import torch.nn as nn


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.
    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SinCosPositionEmbedding(nn.Module):
    '''
    from: patch embeddings
          (batch_size, num_patches, features)
    to:   patch_emb + pos_emb + cls_token
          (batch_size, num_patches + 1, features)
    '''
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.pos_emb = nn.Parameter(
            torch.zeros(1, conf.num_patches + 1, conf.enc.dim),
            requires_grad=False
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, conf.enc.dim),
        )

        self.init_weights()

    def init_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.conf.enc.dim,
            int(self.conf.num_patches ** 0.5),
            add_cls_token=True
        )

        pos_embed = torch.from_numpy(pos_embed)
        pos_embed = pos_embed.float().unsqueeze(0)
        self.pos_emb.data.copy_(pos_embed)

        # class token
        nn.init.normal_(self.cls_token, std=self.conf.init_std)

    def forward(self, x, pri=False):
        batch_size, n_patches, *_ = x.shape
        assert n_patches == self.conf.num_patches

        # add position embeddings w/o cls token
        x = x + self.pos_emb[:, 1:, :]
        if pri: print(x.shape, 'embeddings w/o cls_tokens')

        # append cls token
        cls_token = self.cls_token + self.pos_emb[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        if pri: print(cls_tokens.shape, 'cls_tokens')

        x = torch.cat([cls_tokens, x], dim=1)
        if pri: print(x.shape, 'embeddings w/ cls_tokens')

        return x


# config = Config()
# f = SinCosPositionEmbedding(config)
# x = torch.zeros(2, config.num_patches, config.enc.dim)
# y = f(x, pri=True)
