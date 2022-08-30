from paddle.vision import transforms

def setup_transforms(cfg):
    """
    setup_transforms
    """
    size = cfg.input_size
    tensorize_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(tensorize_transform)
    val_transform = transforms.Compose(tensorize_transform)
    return trn_transform, val_transform