import torch
import torch.nn as nn
from einops import repeat, rearrange

class ShiftViTConfig:
    in_channels = 3
    image_size = (1024, 1024)
    patch_size = (16, 16)

    assert image_size[0] % patch_size[0] == 0
    assert image_size[1] % patch_size[1] == 0
    nh_patches = image_size[0] // patch_size[0]
    nw_patches = image_size[1] // patch_size[1]
    num_patches = nh_patches * nw_patches
    patch_dim = patch_size[0] * patch_size[1] * in_channels

    mask_ratio = 0.75

    class sft:
        n_div = 12
        mlp_ratio = 4
        drop_prob = 0.0
        init_std = 0.02

    class enc(sft):
        dims = [96, 192, 384, 768]
        depth = [2, 2, 6, 6]
        scale_size = [2, 2, 1, None]


    class dec(sft):
        dims = [768, 768, 768]
        depth = [6, 2, 2]
        scale_size = [2, 2, None]




class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=1):
        """ We use GroupNorm (group = 1) to approximate LayerNorm
        for [N, C, H, W] layout"""
        super(GroupNorm, self).__init__(num_groups, num_channels)


class Mlp(nn.Module):

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop_prob=0.):
        """use 1x1 convolution to implement fully-connected MLP layers."""
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ShiftMixer(nn.Module):

    def __init__(self, conf, dim):
        super(ShiftMixer, self).__init__()
        self.conf = conf

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * conf.sft.mlp_ratio)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=dim,
            drop_prob=conf.sft.drop_prob
        )

    def forward(self, x):
        x = self.shift_feat(x, self.conf.sft.n_div)
        shortcut = x
        x = shortcut + self.mlp(self.norm2(x))
        return x

    @staticmethod
    def shift_feat(x, n_div):
        B, C, H, W = x.shape
        g = C // n_div
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out


class PatchMerging(nn.Module):

    def __init__(self, in_dim, out_dim, scale):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim)

        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=(scale, scale),
            stride=scale, bias=False,
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class PatchUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim, scale):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim)

        self.deconv = nn.ConvTranspose2d(
            in_dim, out_dim, kernel_size=(scale, scale),
            stride=scale, bias=False,
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.deconv(x)
        return x


class ShiftViTEncoder(nn.Module):
    def __init__(self, conf):
        super(ShiftViTEncoder, self).__init__()

        blocks = []
        for i in range(len(conf.enc.depth)):
            in_dim = conf.enc.dims[i]
            n_layers = conf.enc.depth[i]
            blocks += nn.ModuleList(
                [ShiftMixer(conf, in_dim)] * n_layers
            )

            scale = conf.enc.scale_size[i]
            if scale is not None:
                out_dim = conf.enc.dims[i + 1]
                blocks.append(PatchMerging(in_dim, out_dim, scale))
        # for i

        self.blocks = nn.ModuleList(blocks)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=conf.sft.init_std)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pri=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if pri: print(x.shape, 'block', i)

        return x

# config = ShiftViTConfig()
# f = ShiftViTEncoder(config)
# x = torch.ones(4, 96, 56, 56)
# y = f(x, pri=True)


class ShiftViTDecoder(nn.Module):
    def __init__(self, conf):
        super(ShiftViTDecoder, self).__init__()

        blocks = []
        for i in range(len(conf.dec.depth)):
            in_dim = conf.dec.dims[i]
            n_layers = conf.dec.depth[i]
            blocks += nn.ModuleList(
                [ShiftMixer(conf, in_dim)] * n_layers
            )

            scale = conf.dec.scale_size[i]
            if scale is not None:
                out_dim = conf.dec.dims[i + 1]
                blocks.append(PatchUpSampling(in_dim, out_dim, scale))
        # for i

        self.blocks = nn.ModuleList(blocks)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=conf.sft.init_std)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pri=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if pri: print(x.shape, 'block', i)

        return x

# config = ShiftViTConfig()
# f = ShiftViTDecoder(config)
# x = torch.ones(4, 768, 16, 16)
# y = f(x, pri=True)


# class PredToken2Image(nn.Module):
#     def __init__(self, conf):
#         super().__init__()
#         self.pred = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(embed_dim, in_chans, (7, 7), (1, 1), (0, 0)),
#             nn.Tanh(),
#         )

class PatchEmbed(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.proj = nn.Conv2d(
            conf.in_channels,
            conf.enc.dims[0],
            kernel_size=conf.patch_size,
            stride=conf.patch_size,
        )

        self.norm = nn.BatchNorm2d(conf.enc.dims[0])

    def forward(self, x, pri=False):
        x = self.proj(x)
        if pri: print(x.shape, 'proj')

        x = self.norm(x)
        return x


# config = ShiftViTConfig()
# f = PatchEmbed(config)
# x = torch.ones(4, 3, 1024, 1024)
# y = f(x, pri=True)


'''
class ShiftViTAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.patch_embed = PatchEmbed(conf)
        self.encoder = ShiftViTEncoder(conf)
        self.decoder = ShiftViTDecoder(conf)

        # self.pred = nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(embed_dim, in_chans, (7, 7), (1, 1), (0, 0)),
        #     nn.Tanh(),
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pri=False):
        if pri: print(x.shape, 'x')

        x = self.patch_embed(x)
        if pri: print(x.shape, 'patch_embed')

        x = self.encoder(x)
        if pri: print(x.shape, 'encoder')

        x = self.decoder(x)
        if pri: print(x.shape, 'decoder')

        x = rearrange(
            x,
            'b (c hp wp) nh nw -> b c (nh hp) (nw wp)',
            hp=self.conf.patch_size[0],
            wp=self.conf.patch_size[1],
        )
        if pri: print(x.shape, 'rearrange')

        # x = self.pred(x)
        # if pri: print(x.shape, 'pred')

        return x

# config = ShiftViTConfig()
# net = ShiftViTAE(config).cuda()
# x = torch.zeros(4,3,1024,1024).cuda()
# y = net(x, True)

'''


class ShiftViTAE(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.patch_embed = PatchEmbed(conf)
        self.encoder = ShiftViTEncoder(conf)
        self.decoder = ShiftViTDecoder(conf)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, conf.enc.dims[0]))
        nn.init.normal_(self.mask_token, std=conf.sft.init_std)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def mask_for(self, x):
        num_masks = int(self.conf.mask_ratio * self.conf.num_patches)

        batch_indices = torch.arange(x.shape[0])
        batch_indices = batch_indices.to(x.device).unsqueeze(-1)

        rand_indices = torch.rand(x.shape[0], self.conf.num_patches)
        rand_indices = rand_indices.to(x.device).argsort(dim=-1)

        masked_indices = rand_indices[:, :num_masks]
        unmasked_indices = rand_indices[:, num_masks:]

        # masked_x = x[batch_indices, masked_indices]
        # unmasked_x = x[batch_indices, unmasked_indices]

        mask_tokens = self.mask_token.expand(x.shape[0], num_masks, -1)

        return {'num_masks': num_masks,
                'batch_indices': batch_indices,
                'masked_indices': masked_indices,
                'unmasked_indices': unmasked_indices,
                'mask_tokens': mask_tokens,
                # 'masked': masked_x,
                # 'unmasked': unmasked_x,
        }

    def forward(self, x, pri=False):
        if pri: print(x.shape, 'x')

        x = self.patch_embed(x)
        if pri: print(x.shape, 'patch_embed')

        assert x.shape[2] == self.conf.nh_patches
        assert x.shape[3] == self.conf.nw_patches
        x = rearrange(x, 'b d nh nw -> b (nh nw) d')
        if pri: print(x.shape, 'x.rearrange')

        mask4emb = self.mask_for(x)
        # n_masks = masked4emb['num_masks']
        batch_idx = mask4emb['batch_indices']
        masked_idx = mask4emb['masked_indices']
        # unmasked_idx = mask4emb['unmasked_indices']
        mask_tokens = mask4emb['mask_tokens']
        if pri: print(mask_tokens.shape, 'mask_tokens')

        x[batch_idx, masked_idx] = mask_tokens
        x = rearrange(x, 'b (nh nw) d -> b d nh nw', nh=self.conf.nh_patches)
        if pri: print(x.shape, 'x.rearrange')

        x = self.encoder(x)
        if pri: print(x.shape, 'encoder')

        x = self.decoder(x)
        if pri: print(x.shape, 'decoder')

        x = rearrange(
            x,
            'b (c hp wp) nh nw -> b c (nh hp) (nw wp)',
            hp=self.conf.patch_size[0],
            wp=self.conf.patch_size[1],
        )
        if pri: print(x.shape, 'rearrange')

        # x = self.pred(x)
        # if pri: print(x.shape, 'pred')

        return x

# config = ShiftViTConfig()
# net = ShiftViTAE(config).cuda()
# x = torch.zeros(4,3,1024,1024).cuda()
# y = net(x, True)






























