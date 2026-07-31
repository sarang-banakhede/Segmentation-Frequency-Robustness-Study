from __future__ import annotations

import numpy as np

try:
    from skimage.morphology import skeletonize as _sk_thin
    def _skeletonize(binary: np.ndarray) -> np.ndarray:
        return _sk_thin(binary.astype(bool))
except ImportError:
    def _skeletonize(binary: np.ndarray) -> np.ndarray:
        img = binary.astype(np.uint8).copy()
        while True:
            P2=img[:-2,1:-1]; P3=img[:-2,2:];  P4=img[1:-1,2:]
            P5=img[2:,2:];   P6=img[2:,1:-1]; P7=img[2:,:-2]
            P8=img[1:-1,:-2]; P9=img[:-2,:-2]
            B  = P2+P3+P4+P5+P6+P7+P8+P9
            A  = (((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)
                 +((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)
                 +((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)
                 +((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8))
            ctr = img[1:-1,1:-1]
            m1  = (ctr==1)&(B>=2)&(B<=6)&(A==1)&(P2*P4*P6==0)&(P4*P6*P8==0)
            img[1:-1,1:-1][m1] = 0
            P2=img[:-2,1:-1]; P3=img[:-2,2:];  P4=img[1:-1,2:]
            P5=img[2:,2:];   P6=img[2:,1:-1]; P7=img[2:,:-2]
            P8=img[1:-1,:-2]; P9=img[:-2,:-2]
            B  = P2+P3+P4+P5+P6+P7+P8+P9
            A  = (((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)
                 +((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)
                 +((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)
                 +((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8))
            ctr = img[1:-1,1:-1]
            m2  = (ctr==1)&(B>=2)&(B<=6)&(A==1)&(P2*P4*P8==0)&(P2*P6*P8==0)
            img[1:-1,1:-1][m2] = 0
            if not m1.any() and not m2.any():
                break
        return img.astype(bool)


def _hd95_assd(pred: np.ndarray, gt: np.ndarray):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return float("nan"), float("nan")
    ps = p & ~binary_erosion(p)
    gs = g & ~binary_erosion(g)
    if not ps.any() or not gs.any():
        return float("nan"), float("nan")
    d_to_g = distance_transform_edt(~g)
    d_to_p = distance_transform_edt(~p)
    d_p2g  = d_to_g[ps]
    d_g2p  = d_to_p[gs]
    hd95   = float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))
    assd   = float((d_p2g.mean() + d_g2p.mean()) / 2.0)
    return hd95, assd


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    from scipy.ndimage import binary_dilation, binary_erosion
    p, g    = pred.astype(bool), gt.astype(bool)
    struct  = np.ones((3, 3), dtype=bool)
    disk    = np.ones((2*tol+1, 2*tol+1), dtype=bool)
    pb = p & ~binary_erosion(p, struct) if p.any() else p
    gb = g & ~binary_erosion(g, struct) if g.any() else g
    if not pb.any() or not gb.any():
        return float("nan")
    eps  = 1e-6
    prec = (pb & binary_dilation(gb, disk)).sum() / (pb.sum() + eps)
    rec  = (gb & binary_dilation(pb, disk)).sum() / (gb.sum() + eps)
    return float(2.0 * prec * rec / (prec + rec + eps))


def _cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return float("nan")
    sp, sg = _skeletonize(p), _skeletonize(g)
    if not sp.sum() or not sg.sum():
        return float("nan")
    eps   = 1e-6
    tprec = (sp & g).sum() / (sp.sum() + eps)
    tsens = (sg & p).sum() / (sg.sum() + eps)
    return float(2.0 * tprec * tsens / (tprec + tsens + eps))


def confusion_counts(pred_bin: np.ndarray, target: np.ndarray):
    p = pred_bin.astype(bool)
    t = target.astype(bool)
    return (float((p &  t).sum()), float((p & ~t).sum()),
            float((~p & t).sum()), float((~p & ~t).sum()))


def metrics_from_confusion(tp: float, fp: float, fn: float, tn: float) -> dict:
    eps  = 1e-6
    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou  = (tp + eps)   / (tp + fp + fn + eps)
    acc  = (tp + tn)    / (tp + tn + fp + fn + eps)
    prec = (tp + eps)   / (tp + fp + eps)
    rec  = (tp + eps)   / (tp + fn + eps)
    spec = (tn + eps)   / (tn + fp + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    fnr  = (fn + eps)   / (fn + tp + eps)
    fpr  = (fp + eps)   / (fp + tn + eps)
    return dict(
        dice_score=dice, iou=iou, pixel_accuracy=acc,
        precision=prec, recall=rec, specificity=spec, f1_score=f1,
        fnr=fnr, fpr=fpr,
    )


def compute_single_metrics(pred_2d: np.ndarray, mask_2d: np.ndarray) -> dict:
    tp, fp, fn, tn = confusion_counts(pred_2d, mask_2d)
    out = metrics_from_confusion(tp, fp, fn, tn)
    h, a = _hd95_assd(pred_2d, mask_2d)
    out["hd95"]     = h
    out["assd"]     = a
    out["bf_score"] = _boundary_f1(pred_2d, mask_2d)
    out["cldice"]   = _cldice(pred_2d, mask_2d)
    return out


def compute_batch_metrics(pred_bin_np: np.ndarray, mask_np: np.ndarray) -> dict:
    tp, fp, fn, tn = confusion_counts(pred_bin_np, mask_np)
    base = metrics_from_confusion(tp, fp, fn, tn)

    B = pred_bin_np.shape[0]
    hd95_l, assd_l, bf_l, cld_l = [], [], [], []
    for i in range(B):
        h, a = _hd95_assd(pred_bin_np[i, 0], mask_np[i, 0])
        hd95_l.append(h)
        assd_l.append(a)
        bf_l.append(_boundary_f1(pred_bin_np[i, 0], mask_np[i, 0]))
        cld_l.append(_cldice(pred_bin_np[i, 0], mask_np[i, 0]))

    def _safe(lst):
        v = [x for x in lst if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    base.update(hd95=_safe(hd95_l), assd=_safe(assd_l),
                bf_score=_safe(bf_l), cldice=_safe(cld_l))
    return base
