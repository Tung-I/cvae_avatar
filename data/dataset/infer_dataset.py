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
from pathlib import Path
from PIL import Image

from data.dataset import BaseDataset
from data.dataset.utils import *


individual = 'm--20181017--0000--002914589--GHS'
target_exp = 'E057_Cheeks_Puffed'
target_cam = '400063'


class InferDataset(BaseDataset):
    def __init__(
        self,
        tex_size=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_dir = '{}/{}'.format(str(self.base_dir), individual)
        self.tex_size = tex_size

        self.target_individual = individual
        self.target_exp = target_exp
        self.target_cam = target_cam

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
    
        files = list(Path("{}/{}".format(self.meshpath, target_exp)).glob("*.bin"))
        frame_nums = [str(f.parts[-1]).split('.')[0] for f in files]
        frame_nums.sort()
        for n in frame_nums:
            avgf = "{}/{}/average/{}.png".format(self.uvpath, target_exp, n)
            if os.path.isfile(avgf) is not True:
                continue
            path = "{}/{}/{}/{}.png".format(self.uvpath, target_exp, target_cam, n)
            if os.path.isfile(path) is True:
                self.framelist.append((target_exp, target_cam, n))

        # compute the view directions
        campos = {}
        extrin = krt[target_cam]["extrin"]
        campos[target_cam] = -np.dot(extrin[:3, :3].T, extrin[:3, 3])
        self.campos = campos

        # load mean image and std
        texmean = np.asarray(
            Image.open("{}/tex_mean.png".format(self.base_dir)), dtype=np.float32
        )
        self.texmean = np.copy(np.flip(texmean, 0))
        self.texmean = cv2.resize(self.texmean, (self.tex_size, self.tex_size))
        self.texstd = float(np.genfromtxt("{}/tex_var.txt".format(self.base_dir)) ** 0.5)
        self.vertmean = np.fromfile(
            "{}/vert_mean.bin".format(self.base_dir), dtype=np.float32
        )
        self.vertstd = float(np.genfromtxt("{}/vert_var.txt".format(self.base_dir)) ** 0.5)



    def __len__(self):
        return len(self.framelist)


    def __getitem__(self, idx):
        exp, cam, frame = self.framelist[idx]
        cam_id = self.camera_ids[cam]

        # geometry
        path = "{}/{}/{}.obj".format(self.meshpath, exp, frame)
        obj = load_obj(path)
        self.mesh_topology = obj

        # geometry
        path = "{}/{}/{}.bin".format(self.meshpath, exp, frame)
        verts = np.fromfile(path, dtype=np.float32)
        verts -= self.vertmean
        verts /= self.vertstd

        # average image
        path = "{}/{}/average/{}.png".format(self.uvpath, exp, frame)
        avgtex = np.asarray(Image.open(path), dtype=np.float32)[::-1, ...]
        mask = avgtex == 0
        avgtex -= self.texmean
        avgtex /= self.texstd
        avgtex[mask] = 0.0
        avgtex = cv2.resize(avgtex, (self.tex_size, self.tex_size)).transpose((2, 0, 1))

        # image
        path = "{}/{}/{}/{}.png".format(self.photopath, exp, cam, frame)
        photo = np.asarray(Image.open(path), dtype=np.float32)
        photo = photo / 255.0

        # texture
        path = "{}/{}/{}/{}.png".format(self.uvpath, exp, cam, frame)
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
            "{}/{}/{}_transform.txt".format(self.meshpath, exp, frame)
        )
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
            "cam": cam_id,
            "M": M.astype(np.float32),
            "uvs": self.mesh_topology["uvs"],
            "vert_ids": self.mesh_topology["vert_ids"],
            "uv_ids": self.mesh_topology["uv_ids"],
            "avg_tex": avgtex,
            "tex": tex,
            "view": view,
            "aligned_verts": verts.reshape((-1, 3)).astype(np.float32),
            "photo": photo
        }