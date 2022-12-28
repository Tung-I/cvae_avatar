import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net.base_net import BaseNet

from model.net.davae_utils import *


class DomainAdaptationVAE2(nn.Module):
    def __init__(
        self,
        im_size=512,
        n_latent=256,
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DomainAdaptationVAE2, self).__init__()
        z_dim = n_latent
        self.enc = DomainAdaptationEncoder(
            im_size, n_latent=z_dim, res=res
        )
        self.dec = DomainAdaptationDecoder(
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


class DomainAdaptationVAE(nn.Module):
    def __init__(
        self,
        im_size=512,
        n_latent=256,
        res=False,
        non=False,
        bilinear=False,
    ):
        super(DomainAdaptationVAE, self).__init__()
        z_dim = n_latent
        self.enc = DomainAdaptationEncoder(
            im_size, n_latent=z_dim, res=res
        )
        self.dec = DomainAdaptationDecoder(
            im_size, z_dim=z_dim, res=res, non=non, bilinear=bilinear
        )
        self.l_map = LinearWN(256, 256)

    def forward(self, up_face, low_face):
        mean, logstd = self.enc(up_face, low_face)
        mean = mean * 0.1
        logstd = logstd * 0.01
        kl = 0.5 * torch.mean(torch.exp(2 * logstd) + mean**2 - 1.0 - 2 * logstd)
        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        z = mean + std * eps
        up_face, low_face = self.dec(z)

        mapped_z = self.l_map(z)

        return up_face, low_face, kl, mapped_z

    def get_model_params(self):
        params = []
        params += list(self.enc.parameters())
        params += list(self.dec.parameters())
        params += list(self.l_map.parameters())
        return params

    # def encode(self, up_face, low_face):
    #     mean, logstd = self.enc(up_face, low_face)
    #     mean = mean * 0.1
    #     logstd = logstd * 0.01
    #     std = torch.exp(logstd)
    #     eps = torch.randn_like(mean)
    #     z = mean + std * eps
    #     mapped_z = self.LinearMapping(z)
    #     return mapped_z


class DomainAdaptationDecoder(nn.Module):
    def __init__(
        self, im_size, z_dim=256, res=False, non=False, bilinear=False
    ):
        super(DomainAdaptationDecoder, self).__init__()
        nhidden = z_dim * 4 * 4 if im_size == 1024 else z_dim * 2 * 2
        self.upface_decoder = TextureDecoder(
            im_size, z_dim, res=res, non=non, bilinear=bilinear
        )
        self.lowface_decoder = TextureDecoder(
            im_size, z_dim, res=res, non=non, bilinear=bilinear
        )
    
        self.z_fc = LinearWN(z_dim, 256)


        self.upface_fc = LinearWN(256, nhidden)
        self.lowface_fc = LinearWN(256, nhidden)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.upface_decoder.upsample[-1].conv2, 1.0)
        glorot(self.lowface_decoder.upsample[-1].conv2, 1.0)

    def forward(self, z):
        z_code = self.relu(self.z_fc(z))
        upface_code = self.relu(self.upface_fc(z_code))
        lowface_code = self.relu(self.lowface_fc(z_code))
        up_face = self.upface_decoder(upface_code)
        low_face = self.lowface_decoder(lowface_code)
        return up_face, low_face


class DomainAdaptationEncoder(nn.Module):
    def __init__(self, inp_size=512, n_latent=256, res=False):
        super(DomainAdaptationEncoder, self).__init__()
        self.n_latent = n_latent
        ntexture_feat = 2048 if inp_size == 1024 else 512
        self.upface_encoder = TextureEncoder(res=res)
        self.lowface_encoder = TextureEncoder(res=res)
        self.upface_fc = LinearWN(ntexture_feat, 256)
        self.lowface_fc = LinearWN(ntexture_feat, 256)
        
        self.fc = LinearWN(512, n_latent * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.fc, 1.0)

    def forward(self, up_face, low_face):
        up_feat = self.relu(self.upface_fc(self.upface_encoder(up_face)))
        low_feat = self.relu(self.lowface_fc(self.lowface_encoder(low_face)))
        feat = torch.cat((up_feat, low_feat), -1)
        latent = self.fc(feat)
        return latent[:, : self.n_latent], latent[:, self.n_latent :]



class DeepAppearanceVAE(nn.Module):
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
        super(DeepAppearanceVAE, self).__init__()
        z_dim = n_latent if mode == "vae" else n_latent * 2
        self.mode = mode
        self.enc = DeepApperanceEncoder(
            tex_size, mesh_inp_size, n_latent=z_dim, res=res
        )
        self.dec = DeepAppearanceDecoder(
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
    


class DeepAppearanceDecoder(nn.Module):
    def __init__(
        self, tex_size, mesh_size, z_dim=128, res=False, non=False, bilinear=False
    ):
        super(DeepAppearanceDecoder, self).__init__()
        nhidden = z_dim * 4 * 4 if tex_size == 1024 else z_dim * 2 * 2
        self.texture_decoder = TextureDecoder(
            tex_size, z_dim, res=res, non=non, bilinear=bilinear
        )
        self.view_fc = LinearWN(3, 8)
        self.z_fc = LinearWN(z_dim, 256)
        self.mesh_fc = LinearWN(256, mesh_size)
        self.texture_fc = LinearWN(256 + 8, nhidden)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.mesh_fc, 1.0)
        glorot(self.texture_decoder.upsample[-1].conv2, 1.0)

    def forward(self, z, v):
        view_code = self.relu(self.view_fc(v))
        z_code = self.relu(self.z_fc(z))
        feat = torch.cat((view_code, z_code), 1)
        texture_code = self.relu(self.texture_fc(feat))
        texture = self.texture_decoder(texture_code)
        mesh = self.mesh_fc(z_code)
        return texture, mesh

    def get_mesh_branch_params(self):
        return list(self.mesh_fc.parameters())

    def get_tex_branch_params(self):
        p = []
        p += list(self.texture_decoder.parameters())
        p += list(self.view_fc.parameters())
        p += list(self.z_fc.parameters())
        p += list(self.texture_fc.parameters())
        return p


class DeepApperanceEncoder(nn.Module):
    def __init__(self, inp_size=1024, mesh_inp_size=21918, n_latent=128, res=False):
        super(DeepApperanceEncoder, self).__init__()
        self.n_latent = n_latent
        ntexture_feat = 2048 if inp_size == 1024 else 512
        self.texture_encoder = TextureEncoder(res=res)
        self.texture_fc = LinearWN(ntexture_feat, 256)
        self.mesh_fc = LinearWN(mesh_inp_size, 256)
        self.fc = LinearWN(512, n_latent * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.fc, 1.0)

    def forward(self, tex, mesh):
        tex_feat = self.relu(self.texture_fc(self.texture_encoder(tex)))
        mesh_feat = self.relu(self.mesh_fc(mesh))
        feat = torch.cat((tex_feat, mesh_feat), -1)
        latent = self.fc(feat)
        return latent[:, : self.n_latent], latent[:, self.n_latent :]

    def get_mesh_branch_params(self):
        return list(self.mesh_fc.parameters())

    def get_tex_branch_params(self):
        p = []
        p += list(self.texture_encoder.parameters())
        p += list(self.texture_fc.parameters())
        p += list(self.fc.parameters())
        return p

