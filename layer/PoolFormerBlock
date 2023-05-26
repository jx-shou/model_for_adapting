import torch
import torch.nn as nn
from einops import repeat, rearrange


class Config:
    in_channels = 3
    image_size = (224, 224)
    patch_size = (4, 4)
    patch_embed_dim = 96
    mask_ratio = 0.75

    class poolformer:
        pool_size = 3
        scale_size = 2

        mlp_ratio = 4
        drop_rate = 0.0
        init_std = 0.02

    class enc(poolformer):
        mode = 'enc'
        dims = [96, 192, 384, 768]
        depth = [2, 2, 6, 2]
        is_scale = [True, True, False, None]

    class dec(poolformer):
        mode = 'dec'
        dims = [768, 384, 192, 96]
        depth = [6, 2, 2, 2]
        is_scale = [True, True, False, None]


class PoolingMixer(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(self, x):
        x = self.pool(x) - x
        return x


# f = Pooling()
# x = torch.ones(2, 96, 56, 56)
# y = f(x)
# print(y.shape)


class ConvMerging(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, conf, di):
        super().__init__()
        is_merge = conf.is_scale[di]
        merge_size = conf.scale_size

        self.proj = nn.Conv2d(
            conf.dims[di],
            conf.dims[di] * 2,
            kernel_size=merge_size if is_merge else 1,
            stride=merge_size if is_merge else 1,
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# f = Merging(96, 192)
# x = torch.ones(2, 96, 56, 56)
# y = f(x)
# print(y.shape)


class ConvUpSampling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, conf, di):
        super().__init__()
        is_up = conf.is_scale[di]
        up_size = conf.scale_size

        self.proj = nn.ConvTranspose2d(
            conf.dims[di],
            conf.dims[di]//2,
            kernel_size=up_size if is_up else 1,
            stride=up_size if is_up else 1,
        )

    def forward(self, x):
        x = self.proj(x)
        return x

# f = Merging(96, 192)
# x = torch.ones(2, 96, 56, 56)
# y = f(x)
# print(y.shape)


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.):
        super().__init__()

        self.fc1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # print(x.shape)

        x = self.fc2(x)
        x = self.drop(x)
        return x

# f = Mlp(96, 192, 96)
# x = torch.ones(2, 96, 56, 56)
# y = f(x)
# print(y.shape)


class UnitBlock(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, conf, dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = PoolingMixer(conf.pool_size)

        self.norm2 = nn.GroupNorm(1, dim)

        hidden_dim = int(dim * conf.mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim, dim, drop=conf.drop_rate)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# f = UnitBlock(96)
# x = torch.ones(2, 96, 56, 56)
# y = f(x)
# print(y.shape)

class PoolFormerNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        blocks = []
        for i in range(len(conf.depth)):
            n_layers = conf.depth[i]
            dim = conf.dims[i]

            unit_list = [UnitBlock(conf, dim)] * n_layers
            blocks += unit_list

            assert conf.mode in ['enc', 'dec']
            if conf.is_scale[i] is not None:
                if conf.mode == 'enc':
                    blocks.append(ConvMerging(conf, i))

                if conf.mode == 'dec':
                    blocks.append(ConvUpSampling(conf, i))

            # if conf.is_scale[i]
        # for i

        self.blocks = nn.ModuleList(blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pri=False):
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if pri: print(x.shape, 'block', idx)

        return x


# config = Config()
# # f = PoolFormerNet(config.enc)
# # x = torch.ones(2, 96, 56, 56)
# # y = f(x, pri=True)
# 
# f = PoolFormerNet(config.dec)
# x = torch.ones(2, 768, 14, 14)
# y = f(x, pri=True)
