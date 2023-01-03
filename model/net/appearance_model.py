import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net.appearance_modules import *


class DeepAppearanceModel(nn.Module):
    """
    Pytorch implementation of "Deep Appearance Models for Face Rendering"
    Args:
        im_size: Size of input images
        tex_size: Size of the predicted unwrapped textures
        mesh_inp_size: Size of the predicted mesh coordinateds (= n_vertex*3)
        n_latent: Feature dimension of the latent space
        n_cams: Maximum number of camera views (for camera color correction)
    """ 
    def __init__(
        self,
        im_size=512,
        tex_size=1024,
        mesh_inp_size=21918,
        n_latent=256,
        n_cams=38,
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DeepAppearanceModel, self).__init__()
        z_dim = n_latent
        self.enc = DomainAdaptiveEncoder(
            im_size, n_latent=z_dim, res=res
        )
        self.mean_map = LinearWN(256, 256)
        self.logstd_map = LinearWN(256, 256)

        self.dec = DeepAvatarDecoder(
            tex_size, mesh_inp_size, z_dim=z_dim, res=res, non=non, bilinear=bilinear
        )
        self.cc = ColorCorrection(n_cams)

    def forward(self, up_face, low_face, view, cams=None):
        b = up_face.size(0)
        latent_mean, latent_logstd = self.enc(up_face, low_face)
        mean = self.mean_map(latent_mean)
        logstd = self.logstd_map(latent_logstd)

        mean = mean * 0.1
        logstd = logstd * 0.01
        kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        z = mean + std * eps

        pred_tex, pred_mesh = self.dec(z, view)
        pred_mesh = pred_mesh.view((b, -1, 3))
        if cams is not None:
            pred_tex = self.cc(pred_tex, cams)
        return pred_tex, pred_mesh, kl



class DomainAdaptiveVAE(nn.Module):
    def __init__(
        self,
        im_size=512,
        n_latent=256,
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DomainAdaptiveVAE, self).__init__()
        z_dim = n_latent
        self.enc = DomainAdaptiveEncoder(
            im_size, n_latent=z_dim, res=res
        )
        self.dec = DomainAdaptiveDecoder(
            im_size, z_dim=z_dim, res=res, non=non, bilinear=bilinear
        )
        self.mean_map = LinearWN(256, 256)
        self.logstd_map = LinearWN(256, 256)

    def forward(self, up_face, low_face):
        latent_mean, latent_logstd = self.enc(up_face, low_face)
        mean = latent_mean * 0.1
        logstd = latent_logstd * 0.01
        kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        up_face, low_face = self.dec(z)

        mapped_mean = self.mean_map(latent_mean)
        mapped_logstd = self.logstd_map(latent_logstd)

        return up_face, low_face, kl, mapped_mean, mapped_logstd

    def get_model_params(self):
        params = []
        params += list(self.enc.parameters())
        params += list(self.dec.parameters())
        params += list(self.mean_map.parameters())
        params += list(self.logstd_map.parameters())
        return params



class DeepAvatarVAE(nn.Module):
    def __init__(
        self,
        tex_size=1024,
        mesh_inp_size=21918,
        mode="vae",
        n_latent=256,
        n_cams=38,
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DeepAvatarVAE, self).__init__()
        z_dim = n_latent if mode == "vae" else n_latent * 2
        self.mode = mode
        self.enc = DeepAvatarEncoder(
            tex_size, mesh_inp_size, n_latent=z_dim, res=res
        )
        self.dec = DeepAvatarDecoder(
            tex_size, mesh_inp_size, z_dim=z_dim, res=res, non=non, bilinear=bilinear
        )
        self.cc = ColorCorrection(n_cams)

    def forward(self, avgtex, mesh, view, cams=None):
        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))
        mean, logstd = self.enc(avgtex, mesh)
        mean = mean * 0.1
        logstd = logstd * 0.01
        if self.mode == "vae":
            kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
            std = torch.exp(logstd)
            eps = torch.randn_like(mean)
            z = mean + std * eps
        else:
            z = torch.cat((mean, logstd), -1)
            kl = torch.tensor(0).to(z.device)

        pred_tex, pred_mesh = self.dec(z, view)
        pred_mesh = pred_mesh.view((b, n, 3))
        if cams is not None:
            pred_tex = self.cc(pred_tex, cams)
        return pred_tex, pred_mesh, kl

    def get_mesh_branch_params(self):
        p = self.enc.get_mesh_branch_params() + self.dec.get_mesh_branch_params()
        return p

    def get_tex_branch_params(self):
        p = self.enc.get_tex_branch_params() + self.dec.get_tex_branch_params()
        return p

    def get_model_params(self):
        params = []
        params += list(self.enc.parameters())
        params += list(self.dec.parameters())
        return params

    def get_cc_params(self):
        return self.cc.parameters()

    def get_latent(self, avgtex, mesh):
        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))
        mean, logstd = self.enc(avgtex, mesh)
        mean = mean * 0.1
        logstd = logstd * 0.01
        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        z = mean + std * eps

        return z

    def encode(self, avgtex, mesh):
        b, n, _ = mesh.shape
        mesh = mesh.view((b, -1))
        mean, logstd = self.enc(avgtex, mesh)

        return mean, logstd
    


