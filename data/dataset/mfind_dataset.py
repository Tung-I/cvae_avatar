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

from utils import *


class MFINDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir='/home/tungi/datasets/mf_individuals/m--20181017--0000--002914589--GHS',
        size=1024
    ):
        self.uvpath = "{}/unwrapped_uv_1024".format(base_dir)
        self.meshpath = "{}/tracked_mesh".format(base_dir)
        self.photopath = "{}/images".format(base_dir)
        self.size = size
        self.krt_path = "{}/KRT".format(base_dir)
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

        # compute view directions of each camera
        campos = {}
        for cam in self.cameras:
            extrin = krt[cam]["extrin"]
            campos[cam] = -np.dot(extrin[:3, :3].T, extrin[:3, 3])
        self.campos = campos

        # load mean image and std
        texmean = np.asarray(
            Image.open("{}/tex_mean.png".format(base_dir)), dtype=np.float32
        )
        self.texmean = np.copy(np.flip(texmean, 0))
        self.texstd = float(np.genfromtxt("{}/tex_var.txt".format(base_dir)) ** 0.5)
        self.texmin = (
            np.zeros_like(self.texmean, dtype=np.float32) - self.texmean
        ) / self.texstd
        self.texmax = (
            np.ones_like(self.texmean, dtype=np.float32) * 255 - self.texmean
        ) / self.texstd

        self.vertmean = np.fromfile(
            "{}/vert_mean.bin".format(base_dir), dtype=np.float32
        )
        self.vertstd = float(np.genfromtxt("{}/vert_var.txt".format(base_dir)) ** 0.5)
        
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
        avgtex = cv2.resize(avgtex, (self.size, self.size)).transpose((2, 0, 1))

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
        tex = cv2.resize(tex, (self.size, self.size)).transpose((2, 0, 1))
        mask = 1.0 - cv2.resize(
            mask.astype(np.float32), (self.size, self.size)
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
            "cam_idx": cam,
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