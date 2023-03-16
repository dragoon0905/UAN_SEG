from .deeplab_multi import DeeplabMulti
from .deeplab import Deeplab


def get_model(cfg):
    if cfg.model.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(lr=cfg.opt.lr)
    
    elif cfg.model.backbone == "deeplabv2":
        model = Deeplab(num_classes=cfg.data.num_classes, init=cfg.model.imagenet_pretrained)
        params = model.optim_parameters(lr=cfg.opt.lr)
    else:
        raise NotImplementedError()
    return model, params
