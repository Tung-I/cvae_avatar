import os
import cv2
import math
import logging
import numpy as np
import imageio
import torch
import json
import random
import sys

from base_dataset import BaseDataset
from utils import *


individual = 'm--20181017--0000--002914589--GHS'
# cam_valid = ['400013', '400042', '400060']


class MFINDDataset(BaseDataset):
    def __init__(
        self,
        tex_size=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_dir = '{}/{}'.format(str(self.base_dir), individual)
        self.tex_size = tex_size

        self.maskpath = "./loss_weight_mask.png"
        self.uvpath = "{}/unwrapped_uv_1024".format(self.base_dir)
        self.meshpath = "{}/tracked_mesh".format(self.base_dir)
        self.photopath = "{}/images".format(self.base_dir)
        self.krt_path = "{}/KRT".format(self.base_dir)
        self.camera_ids = {}
        self.expressions = [str(x.parts[-1]) for x in Path(self.meshpath).iterdir() if x.is_dir()]
        self.expressions.sort()

        check_path(self.uvpath)
        check_path(self.meshpath)
        check_path(self.krt_path)

        # set cameras
        krt = load_krt(self.krt_path)
        self.krt = krt
        self.cameras = list(krt.keys())
        for i, k in enumerate(self.cameras):
            self.camera_ids[k] = i
        self.allcameras = sorted(self.cameras)

        # load train list (but check that images are not dropped!)
        self.framelist = []
        for ep in self.expressions:
            files = list(Path("{}/{}".format(self.meshpath, ep)).glob("*.bin"))
            frame_nums = [str(f.parts[-1]).split('.')[0] for f in files]
            frame_nums.sort()
            for n in frame_nums:
                # check if average texture exists
                avgf = "{}/{}/average/{}.png".format(self.uvpath, ep, n)
                if os.path.isfile(avgf) is not True:
                    continue
                # check if per-view unwrap exists
                for i, cam in enumerate(self.cameras):
                    path = "{}/{}/{}/{}.png".format(self.uvpath, ep, cam, n)
                    if os.path.isfile(path) is True:
                        self.framelist.append((ep, cam, n))

                # for i, cam in enumerate(self.cameras):
                #     path = "{}/{}/{}/{}.png".format(self.uvpath, ep, cam, n)
                #     if os.path.isfile(path) is True:
                #         if self.type == 'train' and cam not in cam_valid:
                #             self.framelist.append((ep, cam, n))
                #         elif self.type =='valid' and cam in cam_valid:
                #             self.framelist.append((ep, cam, n))
                #         else:
                #             continue
                


        # compute view directions of each camera
        campos = {}
        for cam in self.cameras:
            extrin = krt[cam]["extrin"]
            campos[cam] = -np.dot(extrin[:3, :3].T, extrin[:3, 3])
        self.campos = campos

        # load mean image and std
        texmean = np.asarray(
            Image.open("{}/tex_mean.png".format(self.base_dir)), dtype=np.float32
        )
        self.texmean = np.copy(np.flip(texmean, 0))
        self.texstd = float(np.genfromtxt("{}/tex_var.txt".format(self.base_dir)) ** 0.5)
        # self.texmin = (
        #     np.zeros_like(self.texmean, dtype=np.float32) - self.texmean
        # ) / self.texstd
        # self.texmax = (
        #     np.ones_like(self.texmean, dtype=np.float32) * 255 - self.texmean
        # ) / self.texstd

        self.vertmean = np.fromfile(
            "{}/vert_mean.bin".format(self.base_dir), dtype=np.float32
        )
        self.vertstd = float(np.genfromtxt("{}/vert_var.txt".format(self.base_dir)) ** 0.5)

        # weight mask
        self.loss_weight_mask = cv2.flip(cv2.imread(self.maskpath), 0)

        # resize and to_tensor
        self.texmean = cv2.resize(self.texmean, (self.tex_size, self.tex_size))
        self.texmean = torch.tensor(self.texmean).permute((2, 0, 1))[None, ...]
        self.vertmean = torch.tensor(self.vertmean, dtype=torch.float32).view((1, -1, 3))
        self.loss_weight_mask = self.loss_weight_mask / self.loss_weight_mask.max()
        self.loss_weight_mask = torch.tensor(self.loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float()
            
        # sampling for validation and debugging
        if self.type=='valid':
            random.seed(0)
            self.framelist = random.sample(self.framelist, 400)
        if self.debug:
            self.framelist = random.sample(self.framelist, 40)


    def __len__(self):
        return len(self.framelist)


    def __getitem__(self, idx):
        ep, cam, frame = self.framelist[idx]
        cam_id = self.camera_ids[cam]

        # geometry
        if self.mesh_topology is None:
            path = "{}/{}/{}.obj".format(self.meshpath, ep, frame)
            obj = load_obj(path)
            self.mesh_topology = obj

        # geometry
        path = "{}/{}/{}.bin".format(self.meshpath, ep, frame)
        verts = np.fromfile(path, dtype=np.float32)
        verts -= self.vertmean
        verts /= self.vertstd

        # average image
        path = "{}/{}/average/{}.png".format(self.uvpath, ep, frame)
        avgtex = np.asarray(Image.open(path), dtype=np.float32)[::-1, ...]
        mask = avgtex == 0
        avgtex -= self.texmean
        avgtex /= self.texstd
        avgtex[mask] = 0.0
        avgtex = cv2.resize(avgtex, (self.tex_size, self.tex_size)).transpose((2, 0, 1))

        # image
        path = "{}/{}/{}/{}.png".format(self.photopath, ep, cam, frame)
        photo = np.asarray(Image.open(path), dtype=np.float32)
        photo = photo / 255.0

        # texture
        path = "{}/{}/{}/{}.png".format(self.uvpath, ep, cam, frame)
        tex = np.asarray(Image.open(path), dtype=np.float32)[::-1, ...]
        mask = tex == 0
        tex -= self.texmean
        tex /= self.texstd
        tex[mask] = 0.0
        tex = cv2.resize(tex, (self.tex_size, self.tex_size)).transpose((2, 0, 1))
        mask = 1.0 - cv2.resize(
            mask.astype(np.float32), (self.tex_size, self.tex_size)
        ).transpose((2, 0, 1))

        # view direction
        transf = np.genfromtxt(
            "{}/{}/{}_transform.txt".format(self.meshpath, ep, frame)
        )onda install -c conda-forge imgaug
        R_f = transf[:3, :3]
        t_f = transf[:3, 3]
        campos = np.dot(R_f.T, self.campos[cam] - t_f).astype(np.float32)
        view = campos / np.linalg.norm(campos)

        extrin, intrin = self.krt[cam]["extrin"], self.krt[cam]["intrin"]
        R_C = extrin[:3, :3]
        t_C = extrin[:3, 3]
        camrot = np.dot(R_C, R_f).astype(np.float32)
        camt = np.dot(R_C, t_f) + t_C
        camt = camt.astype(np.float32)

        M = intrin @ np.hstack((camrot, camt[None].T))

        return {
            # "cam_idx": cam,
            "frame": frame,
            "exp": ep,
            "cam": cam_id,
            "M": M.astype(np.float32),
            "uvs": self.mesh_topology["uvs"],
            "vert_ids": self.mesh_topology["vert_ids"],
            "uv_ids": self.mesh_topology["uv_ids"],
            "avg_tex": avgtex,
            "mask": mask,
            "tex": tex,
            "view": view,
            "transf": transf.astype(np.float32),
            "aligned_verts": verts.reshape((-1, 3)).astype(np.float32),
            "photo": photo,
        }