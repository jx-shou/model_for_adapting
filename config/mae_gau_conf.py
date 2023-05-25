class Config:
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
    init_std = 0.02

    class enc:
        dim = 768
        mlp_dim = 4 * dim
        n_layers = 12
        layer_norm_eps = 1e-12

    class dec:
        dim = 768
        mlp_dim = 4 * dim
        n_layers = 8
        layer_norm_eps = 1e-12
