import torch.nn as nn

class PatchEmbedding(nn.Module):
    '''
    from: image     (batch_size, channels, height, width)
    to:   embedding (batch_size, num_patches, features)
    '''
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.projection = nn.Conv2d(
            conf.in_channels,
            conf.enc.dim,
            kernel_size=conf.patch_size,
            stride=conf.patch_size,
        )

        self.init_weights()

    def init_weights(self):
        w = self.projection.weight.data
        w = w.view([w.shape[0], -1])
        nn.init.xavier_uniform_(w)

    def forward(self, x, pri=False):
        _, channels, height, width = x.shape
        assert channels == self.conf.in_channels
        assert (height, width) == self.conf.image_size

        x = self.projection(x)
        if pri: print(x.shape, 'projection')

        x = x.flatten(2)
        if pri: print(x.shape, 'x.flatten')

        x = x.transpose(1, 2)
        if pri: print(x.shape, 'x.transpose')

        return x

# config = Config()
# f = PatchEmbeddings(config)
# x = torch.zeros(2, 3, 224, 224)
# y = f(x, pri=True)

