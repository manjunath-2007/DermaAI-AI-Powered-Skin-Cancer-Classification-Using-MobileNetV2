"""Backbone builders. Every model owns its own ImageNet normalisation, so a saved .pt is
self-contained: feed raw RGB in [0,1] and nothing else.

`self.stages` is the ordered list of the torchvision feature-extractor's direct children.
freeze_policy walks it to build layer-wise-LR-decay param groups (last child = deepest).
"""

import torch
import torch.nn as nn

from .common import IMAGENET_MEAN, IMAGENET_STD

WEIGHTS = {
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
    "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights", "IMAGENET1K_V1"),
    "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights", "IMAGENET1K_V1"),
    "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights", "IMAGENET1K_V1"),
    "efficientnet_v2_m": ("efficientnet_v2_m", "EfficientNet_V2_M_Weights", "IMAGENET1K_V1"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    "efficientnet_b3": ("efficientnet_b3", "EfficientNet_B3_Weights", "IMAGENET1K_V1"),
    "swin_t": ("swin_t", "Swin_T_Weights", "IMAGENET1K_V1"),
    "swin_s": ("swin_s", "Swin_S_Weights", "IMAGENET1K_V1"),
}


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BinaryNet(nn.Module):
    """Normalize -> backbone -> GAP -> norm -> dropout -> Linear(1). Returns a raw logit.

    With `aux_classes > 0` an extra 7-way dx head shares the trunk. Fine-grained
    supervision (mel/bcc/akiec vs nv/bkl/vasc/df) is a harder task than the binary one,
    so it acts as a regulariser that sharpens the features the binary head uses.
    """

    def __init__(self, backbone, stages, feat_dim, dropout=0.3, bn_norm=True,
                 channels_last=False, aux_classes=0):
        super().__init__()
        self.pre = Normalize()
        self.backbone = backbone
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm1d(feat_dim) if bn_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.feat_dim = feat_dim
        self.channels_last = channels_last
        self.aux_classes = int(aux_classes)
        self.aux_head = (nn.Linear(feat_dim, self.aux_classes)
                         if self.aux_classes else None)
        if self.aux_head is not None:
            nn.init.normal_(self.aux_head.weight, std=0.01)
            nn.init.zeros_(self.aux_head.bias)

    def features(self, x):
        x = self.pre(x)
        for s in self.stages:
            x = s(x)
        if x.ndim == 4:
            if self.channels_last:
                x = x.permute(0, 3, 1, 2).contiguous()
            x = torch.flatten(self.pool(x), 1)
        elif x.ndim == 3:
            x = x.mean(1)
        return x

    def forward(self, x, return_feat=False):
        f = self.features(x)
        z = self.norm(f)
        logit = self.head(self.dropout(z)).squeeze(1)
        if self.aux_head is not None:
            aux = self.aux_head(z)
            out = (logit, aux) if not return_feat else (logit, f, aux)
        else:
            out = (logit, f) if return_feat else logit
        return out


def build_model(arch="convnext_tiny", pretrained=True, dropout=0.3, bn_norm=True,
                log=None, probe_res=224, aux_classes=0):
    if arch not in WEIGHTS:
        raise SystemExit("FATAL: unknown arch %r (known: %s)" % (arch, ", ".join(WEIGHTS)))
    import torchvision.models as tvm

    fn_name, wcls, went = WEIGHTS[arch]
    fn = getattr(tvm, fn_name)
    weights = None
    if pretrained:
        try:
            weights = getattr(getattr(tvm, wcls), went)
        except AttributeError as exc:
            raise SystemExit("FATAL: no weights enum for %s: %s" % (arch, exc))
    try:
        net = fn(weights=weights)
    except Exception as exc:
        raise SystemExit("FATAL: could not build %s (pretrained=%s): %s\n"
                         "Check the network, or train with pretrained=false." % (arch, pretrained, exc))

    body = net.features
    stages = list(body.children())
    with torch.no_grad():
        was = net.training
        net.eval()
        out = body(torch.zeros(1, 3, probe_res, probe_res))
        net.train(was)
    channels_last = False
    if out.ndim == 4:
        if out.shape[-1] > out.shape[1]:       # NHWC (Swin / MaxViT)
            dim, channels_last = int(out.shape[-1]), True
        else:                                 # NCHW
            dim = int(out.shape[1])
    elif out.ndim == 3:
        dim = int(out.shape[-1])
    else:
        raise SystemExit("FATAL: unexpected feature tensor rank %d for %s" % (out.ndim, arch))

    model = BinaryNet(body, stages, dim, dropout, bn_norm, channels_last, aux_classes)
    if log:
        n = sum(p.numel() for p in model.parameters())
        log("  model    : %-18s pretrained=%-5s dropout=%.2f dim=%d stages=%d aux=%d "
            "params=%.2f M"
            % (arch, pretrained, dropout, dim, len(stages), aux_classes, n / 1e6))
    return model


# --------------------------------------------------------------------------- freezing
def freeze_policy(model, unfreeze, lr_head, lr_backbone, wd, stage_decay=1.0, log=None):
    """Param groups with layer-wise LR decay over the backbone's stage groups.

    unfreeze    : number of trailing stage groups to train (0 = head only, -1 = all).
    stage_decay : per-group LR multiplier; the oldest trained group gets stage_decay**d
                  with d spanning 0 (last group) .. 1 (first trained group).
    """
    groups_bb = list(model.stages)
    n = len(groups_bb)
    uf = n if unfreeze < 0 else max(0, min(unfreeze, n))

    for p in model.parameters():
        p.requires_grad_(False)
    for p in list(model.norm.parameters()) + list(model.head.parameters()):
        p.requires_grad_(True)

    groups = []
    for i, child in enumerate(groups_bb):
        age = n - 1 - i                                  # 0 = last group
        if age >= uf:
            continue
        params = list(child.parameters())
        if not params:
            continue
        for p in params:
            p.requires_grad_(True)
        depth = (age / (uf - 1)) if uf > 1 else 0.0
        groups.append({"params": params, "lr": lr_backbone * (stage_decay ** depth),
                       "name": "s%d" % i, "weight_decay": wd})
    groups.append({"params": list(model.norm.parameters()) + list(model.head.parameters()),
                   "lr": lr_head, "name": "head", "weight_decay": wd})
    if model.aux_head is not None:
        groups.append({"params": list(model.aux_head.parameters()),
                       "lr": lr_head, "name": "aux", "weight_decay": wd})

    if log:
        n_tr = sum(p.numel() for g in groups for p in g["params"])
        n_all = sum(p.numel() for p in model.parameters())
        log("    unfreeze %d/%d stage groups | stage_decay %.2f" % (uf, n, stage_decay))
        for g in groups:
            log("    %-5s lr %.3e  params %9d"
                % (g["name"], g["lr"], sum(p.numel() for p in g["params"])))
        log("  trainable : %d / %d (%.2f%%)" % (n_tr, n_all, 100 * n_tr / n_all))
    return groups