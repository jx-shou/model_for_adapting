import torch.nn as nn

class MaskedEmbeddings(nn.Module):
    '''
    from: image
          (batch_size, channels, height, width)
    to:   embedding w/ cls_token
          (batch_size, num_patches + 1, features)
    '''

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.patch_embed = PatchEmbedding(conf)
        self.pos_embed = SinCosPositionEmbedding(conf)

        self.index_key_set = {
            'num_masks', 'batch_indices',
            'masked_indices', 'unmasked_indices'
        }

    def get_indices_for(self, x):
        num_masks = int(self.conf.mask_ratio * self.conf.num_patches)

        batch_indices = torch.arange(
            x.shape[0],
            device=x.device
        ).unsqueeze(-1)

        rand_indices = torch.rand(
            x.shape[0],
            self.conf.num_patches,
            device=x.device
        ).argsort(dim=-1)

        masked_indices = rand_indices[:, :num_masks]
        unmasked_indices = rand_indices[:, num_masks:]

        return {'num_masks': num_masks,
                'batch_indices': batch_indices,
                'masked_indices': masked_indices,
                'unmasked_indices': unmasked_indices}

    def forward(self, img, indices=None, pri=False):
        patch_emb = self.patch_embed(img)
        if pri: print(patch_emb.shape, 'patch_embed')
        assert patch_emb.shape[1] == self.conf.num_patches

        pos_emb = self.pos_embed(patch_emb)
        cls_tkn = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :]
        if pri: print(cls_tkn.shape, 'cls_token')
        if pri: print(pos_emb.shape, 'pos_embed')

        idx4emb = self.get_indices_for(
            patch_emb,
        ) if indices is None else indices
        assert idx4emb is not None
        assert set(idx4emb.keys()) == self.index_key_set
        batch_idx = idx4emb['batch_indices']
        unmasked_idx = idx4emb['unmasked_indices']

        unmasked_emb = pos_emb[batch_idx, unmasked_idx]
        if pri: print(unmasked_emb.shape, 'unmasked_emb')

        # append cls token
        unmasked_emb = torch.cat([cls_tkn, unmasked_emb], dim=1)
        if pri: print(unmasked_emb.shape, 'unmasked_emb w/ cls_tokens')

        return {'unmasked_emb': unmasked_emb,
                'indices': idx4emb}


# config = Config()
# f = MaskedEmbeddings(config)
# x = torch.ones(2, 3, 224, 224)
# y = f(x, pri=True)
