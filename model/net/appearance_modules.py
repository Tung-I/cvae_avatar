import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptiveDecoder(nn.Module):
    def __init__(
        self, im_size, z_dim=256, res=False, non=False, bilinear=False
    ):
        super(DomainAdaptiveDecoder, self).__init__()
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


class DomainAdaptiveEncoder(nn.Module):
    def __init__(self, inp_size=512, n_latent=256, res=False):
        super(DomainAdaptiveEncoder, self).__init__()
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


class DeepAvatarDecoder(nn.Module):
    def __init__(
        self, tex_size, mesh_size, z_dim=128, res=False, non=False, bilinear=False
    ):
        super(DeepAvatarDecoder, self).__init__()
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


class DeepAvatarEncoder(nn.Module):
    def __init__(self, inp_size=1024, mesh_inp_size=21918, n_latent=128, res=False):
        super(DeepAvatarEncoder, self).__init__()
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



class TextureDecoder(nn.Module):
    def __init__(self, tex_size, z_dim, res=False, non=False, bilinear=False):
        super(TextureDecoder, self).__init__()
        base = 2 if tex_size == 512 else 4
        self.z_dim = z_dim

        self.upsample = nn.Sequential(
            ConvUpsample(
                z_dim, z_dim, 64, base, res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                64, 64, 32, base * (2**2), res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                32, 32, 16, base * (2**4), res=res, use_bilinear=bilinear, non=non
            ),
            ConvUpsample(
                16,
                16,
                3,
                base * (2**6),
                no_activ=True,
                res=res,
                use_bilinear=bilinear,
                non=non,
            ),
        )

    def forward(self, x):
        b, n = x.shape
        h = int(np.sqrt(n / self.z_dim))
        x = x.view((-1, self.z_dim, h, h))
        out = self.upsample(x)
        return out


class TextureEncoder(nn.Module):
    def __init__(self, res=False):
        super(TextureEncoder, self).__init__()
        self.downsample = nn.Sequential(
            ConvDownsample(3, 16, 16, res=res),
            ConvDownsample(16, 32, 32, res=res),
            ConvDownsample(32, 64, 64, res=res),
            ConvDownsample(64, 128, 128, res=res),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.downsample(x)
        out = feat.view((b, -1))
        return out


class MLP(nn.Module):
    def __init__(self, nin, nhidden, nout):
        self.fc1 = LinearWN(nin, nhidden)
        self.fc2 = LinearWN(nhidden, nout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return out


class ConvDownsample(nn.Module):
    def __init__(self, cin, chidden, cout, res=False):
        super(ConvDownsample, self).__init__()
        self.conv1 = Conv2dWN(cin, chidden, 4, 2, padding=1)
        self.conv2 = Conv2dWN(chidden, cout, 4, 2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res = res
        if res:
            self.res1 = Conv2dWN(chidden, chidden, 3, 1, 1)
            self.res2 = Conv2dWN(cout, cout, 3, 1, 1)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        if self.res:
            h = self.relu(self.res1(h) + h)
        h = self.relu(self.conv2(h))
        if self.res:
            h = self.relu(self.res2(h) + h)
        return h


class ConvUpsample(nn.Module):
    def __init__(
        self,
        cin,
        chidden,
        cout,
        feature_size,
        no_activ=False,
        res=False,
        use_bilinear=False,
        non=False,
    ):
        super(ConvUpsample, self).__init__()
        self.conv1 = DeconvTexelBias(
            cin, chidden, feature_size * 2, use_bilinear=use_bilinear, non=non
        )
        self.conv2 = DeconvTexelBias(
            chidden, cout, feature_size * 4, use_bilinear=use_bilinear, non=non
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.no_activ = no_activ
        self.res = res
        if self.res:
            self.res1 = Conv2dWN(chidden, chidden, 3, 1, 1)
            self.res2 = Conv2dWN(cout, cout, 3, 1, 1)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        if self.res:
            h = self.relu(self.res1(h) + h)
        if self.no_activ:
            h = self.conv2(h)
            if self.res:
                h = self.res2(h) + h
        else:
            h = self.relu(self.conv2(h))
            if self.res:
                h = self.relu(self.res2(h) + h)
        return h


class DeconvTexelBias(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        feature_size,
        ksize=4,
        stride=2,
        padding=1,
        use_bilinear=False,
        non=False,
    ):
        super(DeconvTexelBias, self).__init__()
        if isinstance(feature_size, int):
            feature_size = (feature_size, feature_size)
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.deconv = Conv2dWN(cin, cout, 3, 1, 1, bias=False)
        else:
            self.deconv = ConvTranspose2dWN(
                cin, cout, ksize, stride, padding, bias=False
            )
        if non:
            self.bias = nn.Parameter(torch.zeros(1, cout, 1, 1))
        else:
            self.bias = nn.Parameter(
                torch.zeros(1, cout, feature_size[0], feature_size[1])
            )

    def forward(self, x):
        if self.use_bilinear:
            x = F.interpolate(x, scale_factor=2)
        out = self.deconv(x) + self.bias
        return out


class ColorCorrection(nn.Module):
    def __init__(self, n_cameras, nc=3):
        super(ColorCorrection, self).__init__()
        # anchors the 0th camera
        self.weight_anchor = nn.Parameter(torch.ones(1, nc, 1, 1), requires_grad=False)
        self.bias_anchor = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=False)
        self.weight = nn.Parameter(torch.ones(n_cameras - 1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(n_cameras - 1, 3, 1, 1))

    def forward(self, texture, cam):
        weights = torch.cat([self.weight_anchor, self.weight], dim=0)
        biases = torch.cat([self.bias_anchor, self.bias], dim=0)
        w = weights[cam]
        b = biases[cam]
        output = texture * w + b
        return output


def glorot(m, alpha):
    gain = math.sqrt(2.0 / (1.0 + alpha**2))

    if isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return

    # m.weight.data.normal_(0, std)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))
    m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    # if isinstance(m, Conv2dWNUB) or isinstance(m, ConvTranspose2dWNUB) or isinstance(m, LinearWN):
    if (
        isinstance(m, Conv2dWNUB)
        or isinstance(m, Conv2dWN)
        or isinstance(m, ConvTranspose2dWN)
        or isinstance(m, ConvTranspose2dWNUB)
        or isinstance(m, LinearWN)
    ):
        norm = np.sqrt(torch.sum(m.weight.data[:] ** 2))
        m.g.data[:] = norm


class LinearWN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWN, self).__init__(in_features, out_features, bias)
        self.g = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.linear(input, self.weight * self.g[:, None] / wnorm, self.bias)


class Conv2dWNUB(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dWNUB, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return (
            F.conv2d(
                x,
                self.weight * self.g[:, None, None, None] / wnorm,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class Conv2dWN(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.conv2d(
            x,
            self.weight * self.g[:, None, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose2dWNUB(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ConvTranspose2dWNUB, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return (
            F.conv_transpose2d(
                x,
                self.weight * self.g[None, :, None, None] / wnorm,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class ConvTranspose2dWN(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ConvTranspose2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.conv_transpose2d(
            x,
            self.weight * self.g[None, :, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )