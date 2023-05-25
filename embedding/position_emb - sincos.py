import torch.nn as nn

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
