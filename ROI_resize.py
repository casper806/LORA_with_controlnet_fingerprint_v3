import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _largest_component_mask(th):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels <= 1:
        return th

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)

    mask = np.zeros_like(th)
    mask[labels == max_idx] = 255
    return mask


def build_foreground_mask(img):
    """
    Build the fingerprint foreground mask:
    1. Gaussian smoothing
    2. Otsu threshold
    3. Morphological cleanup
    4. Keep the largest connected component
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component_mask(th)


def foreground_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return cx, cy


def foreground_bbox(mask, margin_ratio=0.08):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    mx = max(1, int((x2 - x1) * margin_ratio))
    my = max(1, int((y2 - y1) * margin_ratio))
    return x1 - mx, y1 - my, x2 + mx, y2 + my


def min_enclosing_square(x1, y1, x2, y2, cx, cy, img_w, img_h):
    """Expand a bounding box to a centered square constrained by image bounds."""
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(bw, bh)

    sx1 = int(round(cx - side / 2))
    sy1 = int(round(cy - side / 2))
    sx2 = sx1 + side
    sy2 = sy1 + side

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > img_w:
        sx1 -= sx2 - img_w
        sx2 = img_w
    if sy2 > img_h:
        sy1 -= sy2 - img_h
        sy2 = img_h

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    side = min(sx2 - sx1, sy2 - sy1, img_w - sx1, img_h - sy1)
    side = max(1, side)
    return sx1, sy1, sx1 + side, sy1 + side


def resize_uniform_to_patch(roi, patch_size):
    """Resize a square ROI to patch_size without changing its aspect ratio."""
    side = roi.shape[0]
    if side == patch_size:
        return roi
    interp = cv2.INTER_AREA if side > patch_size else cv2.INTER_CUBIC
    return cv2.resize(roi, (patch_size, patch_size), interpolation=interp)


def extract_patch_from_image(
    img,
    patch_size=512,
    margin_ratio=0.08,
):
    """
    Extract a square foreground ROI and resize it to patch_size.
    Return (patch, was_scaled, was_padded), or (None, False, False) on failure.
    """

    if img is None or img.size == 0:
        return None, False, False

    h, w = img.shape[:2]
    mask = build_foreground_mask(img)

    if np.sum(mask > 0) == 0:
        cx, cy = w // 2, h // 2
        side = min(h, w)
        sx1 = max(0, cx - side // 2)
        sy1 = max(0, cy - side // 2)
        sx2 = min(w, sx1 + side)
        sy2 = min(h, sy1 + side)
        side = min(sx2 - sx1, sy2 - sy1)
        sx2, sy2 = sx1 + side, sy1 + side
    else:
        centroid = foreground_centroid(mask)
        if centroid is None:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = centroid

        bbox = foreground_bbox(mask, margin_ratio=margin_ratio)
        if bbox is None:
            return None, False, False
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        sx1, sy1, sx2, sy2 = min_enclosing_square(x1, y1, x2, y2, cx, cy, w, h)

    roi = img[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None, False, False

    patch = resize_uniform_to_patch(roi, patch_size)
    was_scaled = roi.shape[0] != patch_size
    return patch, was_scaled, False
