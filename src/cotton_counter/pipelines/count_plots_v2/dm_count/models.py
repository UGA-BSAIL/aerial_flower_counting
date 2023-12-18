import contextlib
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    Pose,
    RepC3,
    RepConv,
    RTDETRDecoder,
    Segment,
)
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    colorstr,
    emojis,
    yaml_load,
)
from ultralytics.utils.checks import (
    check_requirements,
    check_suffix,
    check_yaml,
)

try:
    import thop
except ImportError:
    thop = None

__all__ = ["vgg19", "yolov8"]
model_urls = {
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
}


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        # print(mu.size())
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ]
}


def vgg19():
    """VGG 19-layer model (configuration "E")
    model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg["E"]))
    model.load_state_dict(
        model_zoo.load_url(model_urls["vgg19"]), strict=False
    )
    return model


class YOLOv8(nn.Module):
    def __init__(
        self, cfg="yolov8nSPPF-backbone.yaml", ch=3, nc=None, verbose=True
    ):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = (
            cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        )  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(
                f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}"
            )
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError(
                "nc not specified. Must specify nc in model.yaml or function arguments."
            )
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=ch, verbose=verbose
        )  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {
            i: f"{i}" for i in range(self.yaml["nc"])
        }  # default names dict
        self.info()

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(
            nn.Conv2d(64, 1, 1), nn.ReLU()
        )  # 64, 320 only for x

    def forward(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        x = F.upsample_bilinear(x, scale_factor=4)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        # print(mu.size())
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed
        # return x

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(
            self, detailed=detailed, verbose=verbose, imgsz=imgsz
        )


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(
            f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}."
        )
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(
        r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path)
    )  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(
            1
        )  # n, s, m, l, or x
    return ""


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Model information. imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]."""
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.modules()))  # number of layers
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            LOGGER.info(
                "%5g %40s %9s %12g %20s %10.3g %10.3g %10s"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                    p.dtype,
                )
            )

    flops = get_flops(model, imgsz)
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(
        model, "yaml", {}
    ).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(
        f"{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}"
    )
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs."""
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        stride = (
            max(int(model.stride.max()), 32)
            if hasattr(model, "stride")
            else 32
        )  # max stride
        im = torch.empty(
            (1, p.shape[1], stride, stride), device=p.device
        )  # input image in BCHW format
        flops = (
            thop.profile(deepcopy(model), inputs=[im], verbose=False)[0]
            / 1e9
            * 2
            if thop
            else 0
        )  # stride GFLOPs
        imgsz = (
            imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        )  # expand if int/float
        return flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        return 0


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0)
        for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(
                f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'."
            )
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"]
    ):  # + d['head']):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = (
                        locals()[a] if a in locals() else ast.literal_eval(a)
                    )

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
        ):
            c1, c2 = ch[f], args[0]
            if (
                c2 != nc
            ):  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
            ):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            # args =[1, c2, *args[1:]]
            c2 = sum(ch[x] for x in f)
            print(args)
        elif m in (Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif (
            m is RTDETRDecoder
        ):  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}"
            )  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def yolov8():
    model_config = Path(__file__).parent / "yolov8sPAN-backbone.yaml"
    model = YOLOv8(model_config.as_posix())  # yolov8xPAN-TRS-backbone.yaml
    # model.load_state_dict(torch.load('yolov8s.pt'), strict=False)
    return model
