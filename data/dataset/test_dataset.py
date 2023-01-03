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


target_entity = 'm--20181017--0000--002914589--GHS'
input_cam = '400048'
target_exps = ['E008_Smile_Mouth_Closed', 'E019_Frown', 'E057_Cheeks_Puffed', 'E028_Scream_Eyebrows_Up', 'E074_Blink']
target_cam = '400016'


class TestDataset(BaseDataset):
    def __init__(
        self,
        im_size=512,
        tex_size=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_dir = '{}/{}'.format(str(self.base_dir), target_entity)
        self.im_size = im_size
        self.tex_size = tex_size

        self.uvpath = "{}/unwrapped_uv_1024".format(self.base_dir)
        self.meshpath = "{}/tracked_mesh".format(self.base_dir)
        self.photopath = "{}/images".format(self.base_dir)
        self.krt_path = "{}/KRT".format(self.base_dir)
        self.camera_ids = {}

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
        for target_exp in target_exps:
            files = list(Path("{}/{}".format(self.meshpath, target_exp)).glob("*.bin"))
            frame_nums = [str(f.parts[-1]).split('.')[0] for f in files]
            frame_nums.sort()
            for n in frame_nums:
                avgf = "{}/{}/average/{}.png".format(self.uvpath, target_exp, n)
                if os.path.isfile(avgf) is not True:
                    continue
                path = "{}/{}/{}/{}.png".format(self.uvpath, target_exp, target_cam, n)
                if os.path.isfile(path) is not True:
                    continue
                self.framelist.append((target_exp, n))

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
        target_exp, frame = self.framelist[idx]
        cam_id = self.camera_ids[target_cam]

        # target mesh topology
        path = "{}/{}/{}.obj".format(self.meshpath, target_exp, frame)
        obj = load_obj(path)
        self.mesh_topology = obj

        # target mesh vertex coordinates
        path = "{}/{}/{}.bin".format(self.meshpath, target_exp, frame)
        verts = np.fromfile(path, dtype=np.float32)
        verts -= self.vertmean
        verts /= self.vertstd

        # target image
        path = "{}/{}/{}/{}.png".format(self.photopath, target_exp, target_cam, frame)
        photo = np.asarray(Image.open(path), dtype=np.float32)
        photo = photo / 255.0

        # target texture
        path = "{}/{}/{}/{}.png".format(self.uvpath, target_exp, target_cam, frame)
        tex = np.asarray(Image.open(path), dtype=np.float32)[::-1, ...]
        mask = tex == 0
        tex -= self.texmean
        tex /= self.texstd
        tex[mask] = 0.0
        tex = cv2.resize(tex, (self.tex_size, self.tex_size)).transpose((2, 0, 1))
        mask = 1.0 - cv2.resize(
            mask.astype(np.float32), (self.tex_size, self.tex_size)
        ).transpose((2, 0, 1))

        # input view direction
        transf = np.genfromtxt(
            "{}/{}/{}_transform.txt".format(self.meshpath, target_exp, frame)
        )
        R_f = transf[:3, :3]
        t_f = transf[:3, 3]
        campos = np.dot(R_f.T, self.campos[target_cam] - t_f).astype(np.float32)
        view = campos / np.linalg.norm(campos)

        extrin, intrin = self.krt[target_cam]["extrin"], self.krt[target_cam]["intrin"]
        R_C = extrin[:3, :3]
        t_C = extrin[:3, 3]
        camrot = np.dot(R_C, R_f).astype(np.float32)
        camt = np.dot(R_C, t_f) + t_C
        camt = camt.astype(np.float32)

        M = intrin @ np.hstack((camrot, camt[None].T))

        # input image
        path = "{}/{}/{}/{}.png".format(self.photopath, target_exp, input_cam, frame)
        input_photo = np.asarray(Image.open(path), dtype=np.float32) / 255.
        up_face = input_photo[640:640+600, 240:240+600]
        up_face = cv2.resize(up_face, (self.im_size, self.im_size)).transpose((2, 0, 1))
        low_face = input_photo[1024:1024+512, 280:280+512]
        low_face = cv2.resize(low_face, (self.im_size, self.im_size)).transpose((2, 0, 1))


        return {
            "cam": cam_id,
            "M": M.astype(np.float32),
            "uvs": self.mesh_topology["uvs"],
            "vert_ids": self.mesh_topology["vert_ids"],
            "uv_ids": self.mesh_topology["uv_ids"],
            "tex": tex,
            "view": view,
            "gt_verts": verts.reshape((-1, 3)).astype(np.float32),
            "up_face": up_face,
            "low_face": low_face,
            "photo": photo
        }