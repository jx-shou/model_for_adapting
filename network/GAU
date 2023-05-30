import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange


class GAUConfig:
    in_channels = 3
    image_size = (224, 224)
    patch_size = (16, 16)

    assert image_size[0] % patch_size[0] == 0
    assert image_size[1] % patch_size[1] == 0
    nh_patches = image_size[0] // patch_size[0]
    nw_patches = image_size[1] // patch_size[1]
    num_patches = nh_patches * nw_patches
    patch_dim = patch_size[0] * patch_size[1] * in_channels

    mask_ratio = 0.75

    # num_cls = 0
    # assert num_cls in [0, 1]

    class gau:
        norm_eps = 1e-12
        mlp_ratio = 4
        init_std = 0.02
        use_bias = False  # use_offset
        activation = nn.SiLU
        atte_key_size = 128
        atte_norm_list = ['softmax', 'squared_relu', 'softmax_plus']
        atte_normalize = atte_norm_list[2]
        use_atte_scale = True
        drop_rate = 0.1
        atte_drop_rate = 0.1
        inf = 1e4

    class enc(gau):
        dim = 768
        n_layers = 12

    class dec(gau):
        dim = 768
        n_layers = 8

def attention_normalize(a, mask=None, dim=-1, method=None):
    """不同的注意力归一化方案
    softmax：常规/标准的指数归一化；
    squared_relu：来自 https://arxiv.org/abs/2202.10447 ；
    softmax_plus：来自 https://kexue.fm/archives/8823 。
    """
    assert method in ['softmax', 'squared_relu', 'softmax_plus']

    if method == "softmax":
        return torch.softmax(a, dim=dim)
    else:
        if mask is not None:
            assert mask.ndim == 3
            ll = mask.sum(-1, keepdim=True)
        else:
            ll = torch.ones_like(a) * a.shape[-2]
        if method == "squared_relu":
            return torch.relu(a) ** 2 / ll
        elif method == "softmax_plus":
            scale = torch.log(ll) / np.log(512)
            # mask: 1 for not padding, 0 for padding
            # padding position's scale is 1
            if mask is not None:
                scale = scale.masked_fill(mask == 0, 1.0)
            return torch.softmax(a * scale, dim=dim)
    return a


class ScaleOffset(nn.Module):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
    """

    def __init__(
        self,
        hidden_size=768,
        scale=True,
        offset=True,
    ):
        super().__init__()
        self.scale = scale
        self.offset = offset

        if self.scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))

        if self.offset:
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs):
        if self.scale:
            inputs = inputs * self.weight

        if self.offset:
            inputs = inputs + self.bias

        return inputs


class Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


class GatedAttentionUnit(nn.Module):
    """门控注意力单元
    链接：https://arxiv.org/abs/2202.10447
    介绍：https://kexue.fm/archives/8934
    说明：没有加入加性相对位置编码，个人认为是不必要的；如果觉得有必要，
         可以自行通过a_bias传入。
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.act = conf.activation()

        i_hidden_dim = conf.dim * conf.mlp_ratio + conf.atte_key_size
        o_hidden_dim = conf.dim * conf.mlp_ratio // 2
        self.i_dense = nn.Linear( conf.dim, i_hidden_dim, bias=conf.use_bias)
        self.o_dense = nn.Linear(o_hidden_dim, conf.dim, bias=conf.use_bias)

        self.q_scaleoffset = ScaleOffset(conf.atte_key_size, offset=conf.use_bias)
        self.k_scaleoffset = ScaleOffset(conf.atte_key_size, offset=conf.use_bias)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos=None):
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos
        # x.shape [batch, seq_len, 2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # [cos_nθ, -sin_nθ] [x1]
        # [sin_nθ,  cos_nθ] [x2]
        # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
        # 苏神的rotary，使用了下面的计算方法。
        # return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2, -1)
        # 考虑到矩阵乘法torch.einsum("bmd,bnd->bmn", q, k)，因此可以直接在最后一个维度拼接（无需奇偶交错）
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        output_attentions=False,
    ):
        # 投影变换
        x = self.i_dense(hidden_states)
        x = self.act(x)
        u, v, qk = torch.split(
            x,
            [
                self.conf.dim * self.conf.mlp_ratio // 2,
                self.conf.dim * self.conf.mlp_ratio // 2,
                self.conf.atte_key_size
            ],
            dim=-1,
        )

        q = self.q_scaleoffset(qk)
        k = self.k_scaleoffset(qk)

        # 加入RoPE
        q = self.apply_rotary(q, sinusoidal_pos)
        k = self.apply_rotary(k, sinusoidal_pos)

        # Attention
        a = torch.einsum("bmd,bnd->bmn", q, k)

        if self.conf.use_atte_scale:
            a = a / self.conf.atte_key_size ** 0.5

        if attention_mask is not None:
            a = a.masked_fill(attention_mask == 0, -self.conf.inf)

        A = attention_normalize(
            a, attention_mask, dim=-1,
            method=self.conf.atte_normalize
        )

        A = F.dropout(A, p=self.conf.atte_drop_rate, training=self.training)

        # 计算输出
        out = self.o_dense(u * torch.einsum("bmn,bnd->bmd", A, v))

        # outputs = (o, A) if output_attentions else (o,)
        # return outputs
        return out


class GAULayer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.gau = GatedAttentionUnit(conf)
        self.norm = Norm(eps=conf.norm_eps)

    def forward(
        self, hidden_states, attention_mask=None,
        sinusoidal_pos=None, output_attentions=False,
    ):
        # 投影变换
        gau_output = self.gau(
            hidden_states, attention_mask, sinusoidal_pos, output_attentions
        )

        # dropout and residual
        o = F.dropout(
            gau_output, p=self.conf.drop_rate, training=self.training
        )
        o = self.norm(hidden_states + o)

        # outputs = (o,) + gau_output[1:]  # add attentions if we output them
        # return outputs
        return o


# f = GAULayer()
# x = torch.zeros(2, 192, 768)
# y = f(x)
# print(y[0].shape)




class GAUEncoder(nn.Module):
    '''
    from: embeddings
          (batch_size, num_patches, features)
    to:   embeddings
          (batch_size, num_patches, features)
    '''
    def __init__(self, conf):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GAULayer(conf.enc) for _ in range(conf.enc.n_layers)]
            )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=conf.enc.init_std)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, emb, pri=False):
        for idx, block in enumerate(self.blocks):
            emb = block(emb)
            if pri: print(emb.shape, 'enc_block', idx)

        return emb


# config = GAUConfig()
# f = GAUEncoder(config)
# x = torch.ones(2, 192, config.enc.dim)
# y = f(x, pri=True)


class GAUToken2Image(nn.Module):
    '''
    from: embeddings
          (batch_size, num_patches, features)
    to:   image
          (batch_size, channels, height, width)
    '''
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.pred = nn.Linear(conf.dec.dim, conf.patch_dim, bias=True)


    def forward(self, emb, pri=False):
        emb = self.pred(emb)
        if pri: print(emb.shape, 'pred')

        emb = rearrange(
            emb,
            'b (nh nw) (hp wp c) -> b c (nh hp) (nw wp)',
            nh=self.conf.nh_patches,
            hp=self.conf.patch_size[0],
            wp=self.conf.patch_size[1],
        )
        if pri: print(emb.shape, 'emb.rearrange')

# config = GAUConfig()
# f = GAUToken2Image(config)
# x = torch.ones(2, 196, config.dec.dim)
# y = f(x, pri=True)

class GAUDecoder(nn.Module):
    '''
    from: embeddings
          (batch_size, num_patches, features)
    to:   embeddings
          (batch_size, num_patches, features)
    '''
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.enc2dec = nn.Linear(
            conf.enc.dim, conf.dec.dim,
        ) if conf.enc.dim != conf.dec.dim else nn.Identity()

        self.blocks = nn.ModuleList(
            [GAULayer(conf.dec) for _ in range(conf.dec.n_layers)]
        )

        self.dec_norm = nn.LayerNorm(conf.dec.dim, eps=conf.dec.norm_eps)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=conf.dec.init_std)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, emb, pri=False):
        emb = self.enc2dec(emb)
        if pri: print(emb.shape, 'enc2dec')

        for idx, block in enumerate(self.blocks):
            emb = block(emb)
            if pri: print(emb.shape, 'dec_block', idx)

        emb = self.dec_norm(emb)
        if pri: print(emb.shape, 'emb.norm')

        return emb


# config = GAUConfig()
# f = GAUDecoder(config)
# x = torch.ones(2, 196, config.dec.dim)
# y = f(x, pri=True)

