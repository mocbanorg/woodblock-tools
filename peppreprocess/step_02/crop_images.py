from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class LocalizationResult:
    bbox: Tuple[int, int, int, int]  # low-res x0, y0, x1, y1
    method: str
    confidence: float
    debug_core: np.ndarray


def odd(v: float, minimum: int = 3) -> int:
    n = max(minimum, int(round(v)))
    return n if n % 2 == 1 else n + 1


def downscale(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale == 1.0:
        return image.copy(), 1.0
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale


def border_background_lab(lab: np.ndarray, frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray, int]:
    h, w = lab.shape[:2]
    b = max(8, int(min(h, w) * frac))
    border = np.concatenate(
        [
            lab[:b].reshape(-1, 3),
            lab[-b:].reshape(-1, 3),
            lab[:, :b].reshape(-1, 3),
            lab[:, -b:].reshape(-1, 3),
        ],
        axis=0,
    )
    median = np.median(border, axis=0)
    return median.astype(np.float32), border.astype(np.float32), b


def compute_features(small: np.ndarray):
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bg_med, bg_border, b = border_background_lab(lab)

    delta = lab - bg_med
    # L giảm trọng số một chút để bền với thay đổi chiếu sáng.
    color_dist = np.sqrt((delta[..., 0] / 1.6) ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    tex_k = odd(min(gray.shape) * 0.020, 15)
    texture = cv2.blur(grad, (tex_k, tex_k))

    border_dist = np.concatenate(
        [color_dist[:b].ravel(), color_dist[-b:].ravel(), color_dist[:, :b].ravel(), color_dist[:, -b:].ravel()]
    )
    border_tex = np.concatenate(
        [texture[:b].ravel(), texture[-b:].ravel(), texture[:, :b].ravel(), texture[:, -b:].ravel()]
    )
    return lab, gray, color_dist, texture, gx, gy, bg_med, border_dist, border_tex, b


def component_score(stats, centroid, shape, image_shape) -> float:
    x, y, w, h, area = [float(v) for v in stats]
    H, W = image_shape
    img_area = H * W
    cx, cy = centroid
    center_dist = math.sqrt(((cx - W / 2) / (W / 2)) ** 2 + ((cy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    centrality = max(0.0, 1.0 - center_dist)
    fill = area / max(1.0, w * h)
    aspect = w / max(h, 1.0)
    # Mộc bản thường gần chữ nhật và không phải dải rất mỏng như thước.
    aspect_ok = min(1.0, aspect / 1.0) if aspect < 1.0 else min(1.0, 5.0 / aspect)
    return 6.0 * (area / img_area) + 1.8 * centrality + 0.5 * min(fill, 0.95) + 0.4 * aspect_ok


def best_component_bbox(mask: np.ndarray, min_area_ratio: float = 0.006):
    H, W = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < H * W * min_area_ratio:
            continue
        score = component_score(stats[i], centroids[i], labels, (H, W))
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    i = best[1]
    x, y, w, h, area = stats[i]
    return (int(x), int(y), int(x + w), int(y + h)), float(best[0]), (labels == i).astype(np.uint8) * 255


def texture_bbox(texture: np.ndarray, border_tex: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    H, W = texture.shape
    # Giữ vùng texture cao: chữ khắc tạo mật độ biên lớn hơn nền, bóng đổ, giấy/thước.
    thr = max(float(np.percentile(texture, 72)), float(np.percentile(border_tex, 92) * 1.08))
    mask = (texture > thr).astype(np.uint8) * 255

    open_k = odd(min(H, W) * 0.003)
    close_x = odd(W * 0.035, 9)
    close_y = odd(H * 0.035, 9)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_x, close_y)))

    result = best_component_bbox(mask, min_area_ratio=0.005)
    if result is None:
        return None
    bbox, score, core = result
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    area_ratio = bw * bh / float(H * W)
    # Loại các component quá nhỏ / quá dị thường.
    if bw < 0.18 * W or bh < 0.12 * H or area_ratio > 0.90:
        return None

    # Texture core nằm hơi lọt vào trong mộc bản => nới bbox để tìm cạnh thật ở bước sau.
    mx = int(0.10 * bw)
    my = int(0.22 * bh)
    rough = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
    confidence = min(1.0, 0.45 + area_ratio + 0.05 * score)
    return rough, confidence, core


def color_bbox(lab: np.ndarray, color_dist: np.ndarray, bg_med: np.ndarray, border_dist: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    H, W = color_dist.shape
    dthr = max(13.0, float(np.percentile(border_dist, 80) + 8.0))
    dark = lab[..., 0] < (bg_med[0] - 20.0)
    mask = ((color_dist > dthr) | dark).astype(np.uint8) * 255

    edge = max(2, int(min(H, W) * 0.004))
    mask[:edge] = 0
    mask[-edge:] = 0
    mask[:, :edge] = 0
    mask[:, -edge:] = 0

    op = odd(min(H, W) * 0.004)
    cl = odd(min(H, W) * 0.012, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op, op)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl, cl)))

    result = best_component_bbox(mask, min_area_ratio=0.02)
    if result is None:
        return None
    bbox, score, core = result
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    area_ratio = bw * bh / float(H * W)
    if bw < 0.18 * W or bh < 0.12 * H or area_ratio > 0.92:
        return None
    mx = int(0.025 * bw)
    my = int(0.035 * bh)
    rough = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
    confidence = min(1.0, 0.40 + area_ratio + 0.04 * score)
    return rough, confidence, core



def dark_bbox(lab: np.ndarray, bg_med: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
    """Specialized locator for bright/white backgrounds: keep the large dark board, ignore rulers/labels."""
    H, W = lab.shape[:2]
    # Relative threshold handles exposure changes; absolute cap avoids swallowing light background.
    lthr = min(185.0, float(bg_med[0] - 30.0))
    mask = (lab[..., 0] < lthr).astype(np.uint8) * 255
    edge = max(2, int(min(H, W) * 0.003))
    mask[:edge] = 0; mask[-edge:] = 0; mask[:, :edge] = 0; mask[:, -edge:] = 0
    op = odd(min(H, W) * 0.003)
    cl = odd(min(H, W) * 0.008, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op, op)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl, cl)))
    result = best_component_bbox(mask, min_area_ratio=0.02)
    if result is None:
        return None
    bbox, score, core = result
    x0, y0, x1, y1 = bbox
    bw, bh = x1-x0, y1-y0
    ar = bw*bh/float(H*W)
    if bw < 0.18*W or bh < 0.12*H or ar > 0.88:
        return None
    mx, my = int(0.02*bw), int(0.03*bh)
    rough = (max(0,x0-mx), max(0,y0-my), min(W,x1+mx), min(H,y1+my))
    conf = min(1.0, 0.55 + ar + 0.04*score)
    return rough, conf, core

def _side_score_top(y, x0, x1, texture, color_dist, gy_abs, delta) -> float:
    H = texture.shape[0]
    a0, a1 = max(0, y - 3 * delta), max(0, y - delta)
    b0, b1 = min(H, y + delta), min(H, y + 3 * delta)
    if a1 <= a0 or b1 <= b0:
        return -1e9
    xs0 = x0 + int(0.08 * (x1 - x0))
    xs1 = x1 - int(0.08 * (x1 - x0))
    out_tex = float(texture[a0:a1, xs0:xs1].mean())
    in_tex = float(texture[b0:b1, xs0:xs1].mean())
    out_col = float(color_dist[a0:a1, xs0:xs1].mean())
    in_col = float(color_dist[b0:b1, xs0:xs1].mean())
    edge = float(gy_abs[max(0, y - 1):min(H, y + 2), xs0:xs1].mean())
    gain = max(0.0, (in_tex - out_tex) / (out_tex + 1e-3)) + 0.5 * max(0.0, (in_col - out_col) / (out_col + 1e-3))
    return edge * (1.0 + 1.8 * gain)


def _side_score_bottom(y, x0, x1, texture, color_dist, gy_abs, delta) -> float:
    H = texture.shape[0]
    a0, a1 = max(0, y - 3 * delta), max(0, y - delta)  # inside
    b0, b1 = min(H, y + delta), min(H, y + 3 * delta)  # outside
    if a1 <= a0 or b1 <= b0:
        return -1e9
    xs0 = x0 + int(0.08 * (x1 - x0))
    xs1 = x1 - int(0.08 * (x1 - x0))
    in_tex = float(texture[a0:a1, xs0:xs1].mean())
    out_tex = float(texture[b0:b1, xs0:xs1].mean())
    in_col = float(color_dist[a0:a1, xs0:xs1].mean())
    out_col = float(color_dist[b0:b1, xs0:xs1].mean())
    edge = float(gy_abs[max(0, y - 1):min(H, y + 2), xs0:xs1].mean())
    gain = max(0.0, (in_tex - out_tex) / (out_tex + 1e-3)) + 0.5 * max(0.0, (in_col - out_col) / (out_col + 1e-3))
    return edge * (1.0 + 1.8 * gain)


def _side_score_left(x, y0, y1, texture, color_dist, gx_abs, delta) -> float:
    W = texture.shape[1]
    a0, a1 = max(0, x - 3 * delta), max(0, x - delta)
    b0, b1 = min(W, x + delta), min(W, x + 3 * delta)
    if a1 <= a0 or b1 <= b0:
        return -1e9
    ys0 = y0 + int(0.08 * (y1 - y0))
    ys1 = y1 - int(0.08 * (y1 - y0))
    out_tex = float(texture[ys0:ys1, a0:a1].mean())
    in_tex = float(texture[ys0:ys1, b0:b1].mean())
    out_col = float(color_dist[ys0:ys1, a0:a1].mean())
    in_col = float(color_dist[ys0:ys1, b0:b1].mean())
    edge = float(gx_abs[ys0:ys1, max(0, x - 1):min(W, x + 2)].mean())
    gain = max(0.0, (in_tex - out_tex) / (out_tex + 1e-3)) + 0.5 * max(0.0, (in_col - out_col) / (out_col + 1e-3))
    return edge * (1.0 + 1.8 * gain)


def _side_score_right(x, y0, y1, texture, color_dist, gx_abs, delta) -> float:
    W = texture.shape[1]
    a0, a1 = max(0, x - 3 * delta), max(0, x - delta)  # inside
    b0, b1 = min(W, x + delta), min(W, x + 3 * delta)  # outside
    if a1 <= a0 or b1 <= b0:
        return -1e9
    ys0 = y0 + int(0.08 * (y1 - y0))
    ys1 = y1 - int(0.08 * (y1 - y0))
    in_tex = float(texture[ys0:ys1, a0:a1].mean())
    out_tex = float(texture[ys0:ys1, b0:b1].mean())
    in_col = float(color_dist[ys0:ys1, a0:a1].mean())
    out_col = float(color_dist[ys0:ys1, b0:b1].mean())
    edge = float(gx_abs[ys0:ys1, max(0, x - 1):min(W, x + 2)].mean())
    gain = max(0.0, (in_tex - out_tex) / (out_tex + 1e-3)) + 0.5 * max(0.0, (in_col - out_col) / (out_col + 1e-3))
    return edge * (1.0 + 1.8 * gain)


def refine_bbox_by_edges(rough, core, texture, color_dist, gx, gy) -> Tuple[int, int, int, int]:
    H, W = texture.shape
    x0, y0, x1, y1 = rough
    bw, bh = x1 - x0, y1 - y0
    if bw < 20 or bh < 20:
        return rough

    gx_abs, gy_abs = np.abs(gx), np.abs(gy)
    dy = max(2, int(0.010 * bh))
    dx = max(2, int(0.010 * bw))

    # Texture/color core thường nằm bên TRONG tấm gỗ. Tìm cạnh thật quanh core,
    # thay vì quét cả rough bbox; điều này tránh chọn nhầm cạnh thước ở phía trên/trái.
    ys, xs = np.where(core > 0)
    if len(xs) > 0:
        cx0, cx1 = int(xs.min()), int(xs.max()) + 1
        cy0, cy1 = int(ys.min()), int(ys.max()) + 1
        cw, ch = max(1, cx1 - cx0), max(1, cy1 - cy0)
        tx0 = max(x0 + dx * 3, cx0 - int(0.28 * cw))
        tx1 = min(x1 - dx * 3, cx0 + int(0.10 * cw))
        bx0 = max(x0 + dx * 3, cx1 - int(0.10 * cw))
        bx1 = min(x1 - dx * 3, cx1 + int(0.28 * cw))
        ty0 = max(y0 + dy * 3, cy0 - int(0.32 * ch))
        ty1 = min(y1 - dy * 3, cy0 + int(0.10 * ch))
        by0 = max(y0 + dy * 3, cy1 - int(0.10 * ch))
        by1 = min(y1 - dy * 3, cy1 + int(0.32 * ch))
        top_range = range(ty0, max(ty0 + 1, ty1))
        bot_range = range(by0, max(by0 + 1, by1))
        left_range = range(tx0, max(tx0 + 1, tx1))
        right_range = range(bx0, max(bx0 + 1, bx1))
    else:
        top_range = range(y0 + dy * 3, max(y0 + dy * 4, y0 + int(0.42 * bh)))
        bot_range = range(min(y1 - dy * 4, y0 + int(0.58 * bh)), y1 - dy * 3)
        left_range = range(x0 + dx * 3, max(x0 + dx * 4, x0 + int(0.42 * bw)))
        right_range = range(min(x1 - dx * 4, x0 + int(0.58 * bw)), x1 - dx * 3)

    def argbest(it, fn, fallback):
        vals = [(fn(v), v) for v in it]
        vals = [t for t in vals if np.isfinite(t[0])]
        return max(vals)[1] if vals else fallback

    top = argbest(top_range, lambda y: _side_score_top(y, x0, x1, texture, color_dist, gy_abs, dy), y0)
    bottom = argbest(bot_range, lambda y: _side_score_bottom(y, x0, x1, texture, color_dist, gy_abs, dy), y1 - 1)
    left = argbest(left_range, lambda x: _side_score_left(x, y0, y1, texture, color_dist, gx_abs, dx), x0)
    right = argbest(right_range, lambda x: _side_score_right(x, y0, y1, texture, color_dist, gx_abs, dx), x1 - 1)

    # Nếu refinement tạo box phi lý thì giữ rough.
    if right - left < 0.45 * bw or bottom - top < 0.45 * bh:
        return rough

    pad_x = max(2, int(0.006 * (right - left)))
    pad_y = max(2, int(0.008 * (bottom - top)))
    return max(0, left - pad_x), max(0, top - pad_y), min(W, right + 1 + pad_x), min(H, bottom + 1 + pad_y)


def localize(small: np.ndarray) -> Tuple[LocalizationResult, dict]:
    lab, gray, dist, texture, gx, gy, bg_med, border_dist, border_tex, border_band = compute_features(small)
    t = texture_bbox(texture, border_tex)
    c = color_bbox(lab, dist, bg_med, border_dist)
    d = dark_bbox(lab, bg_med) if bg_med[0] >= 180.0 else None

    # Nền sáng/trắng: dark mask tách tấm gỗ khỏi thước/nhãn tốt hơn color-distance.
    # Nền xanh/low-contrast (nhóm MB): ưu tiên texture của chữ khắc.
    if d is not None and bg_med[0] >= 180.0:
        rough, conf, core, method = d[0], d[1], d[2], "dark"
    elif c is not None and bg_med[0] >= 188.0:
        rough, conf, core, method = c[0], c[1], c[2], "color"
    elif t is not None:
        rough, conf, core = t
        method = "texture"
        # Nếu texture cho box cực nhỏ còn color hợp lý hơn, ưu tiên color.
        if c is not None:
            cb, cc, ccore = c
            ta = (rough[2] - rough[0]) * (rough[3] - rough[1])
            ca = (cb[2] - cb[0]) * (cb[3] - cb[1])
            if ta < 0.45 * ca and cc > conf + 0.15:
                rough, conf, core, method = cb, cc, ccore, "color"
    elif c is not None:
        rough, conf, core, method = c[0], c[1], c[2], "color"
    else:
        H, W = small.shape[:2]
        margin_x, margin_y = int(0.03 * W), int(0.03 * H)
        rough = (margin_x, margin_y, W - margin_x, H - margin_y)
        core = np.zeros((H, W), np.uint8)
        conf, method = 0.0, "fallback_center"

    refined = refine_bbox_by_edges(rough, core, texture, dist, gx, gy)
    result = LocalizationResult(refined, method + "+edge", conf, core)
    feats = {
        "lab": lab,
        "dist": dist,
        "texture": texture,
        "bg_med": bg_med,
        "border_dist": border_dist,
        "border_tex": border_tex,
    }
    return result, feats


def cleanup_foreground_mask(small: np.ndarray, bbox, feats: dict) -> np.ndarray:
    """Mask nền chỉ trong bbox. Xóa nền giống border nếu nó thông ra mép crop."""
    x0, y0, x1, y1 = bbox
    dist = feats["dist"][y0:y1, x0:x1]
    tex = feats["texture"][y0:y1, x0:x1]
    border_dist = feats["border_dist"]
    border_tex = feats["border_tex"]

    bg_d = max(11.0, float(np.percentile(border_dist, 86) + 4.0))
    bg_t = max(float(np.percentile(border_tex, 88) * 1.20), float(np.median(border_tex) + 3.0))
    bg_like = ((dist < bg_d) & (tex < bg_t)).astype(np.uint8)

    # Component background-like nào chạm mép crop thì coi là nền thật.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg_like, 8)
    remove = np.zeros_like(bg_like, dtype=np.uint8)
    H, W = bg_like.shape
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
            remove[labels == i] = 1

    fg = (1 - remove).astype(np.uint8) * 255
    # Giữ component lớn, ưu tiên component chứa tâm crop.
    op = odd(min(H, W) * 0.003)
    cl = odd(min(H, W) * 0.010, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (op, op)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl, cl)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return np.ones_like(fg) * 255
    center = np.array([W / 2.0, H / 2.0])
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.03 * H * W:
            continue
        c = centroids[i]
        central = 1.0 - np.linalg.norm((c - center) / np.maximum(center, 1.0)) / math.sqrt(2)
        score = area / (H * W) + 0.25 * central
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return fg
    main = (labels == best[1]).astype(np.uint8) * 255

    # Chỉ fill các lỗ KÍN bên trong object; không dùng fill toàn contour vì cách đó
    # có thể kéo nhãn giấy / tam giác nền ở cạnh vào lại mask.
    padded = cv2.copyMakeBorder(main, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ffmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)[1:-1, 1:-1]
    return filled



def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)  # y - x
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def rectify_perspective(board_crop: np.ndarray):
    """Tìm đường viền ngoài của mộc bản trong crop và hiệu chỉnh về hình chữ nhật.

    Nếu không tìm được tứ giác đủ tin cậy, trả lại ảnh gốc để tránh biến dạng sai.
    Phù hợp nhất khi mộc bản tối màu, nền sáng và đã được crop tương đối sát.
    """
    if board_crop is None or board_crop.size == 0:
        return board_crop, False, None, None, None

    gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY) if board_crop.ndim == 3 else board_crop.copy()
    h, w = gray.shape[:2]
    if min(h, w) < 40:
        return board_crop, False, None, None, None

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Mộc bản thường tối hơn nền; Otsu tách vùng tối, sau đó nối các đoạn cạnh bị đứt.
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, int(round(min(h, w) * 0.012)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return board_crop, False, None, None, None

    # Bỏ contour quá nhỏ; contour mộc bản thường chiếm phần lớn crop.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = next((c for c in contours if cv2.contourArea(c) >= 0.15 * h * w), None)
    if contour is None:
        return board_crop, False, None, None, None

    perimeter = cv2.arcLength(contour, True)
    quad = None
    for eps_ratio in (0.012, 0.018, 0.025, 0.035, 0.045):
        approx = cv2.approxPolyDP(contour, eps_ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return board_crop, False, None, None, None

    src = order_quad_points(quad)
    tl, tr, br, bl = src
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_w = int(round(max(width_top, width_bottom)))
    out_h = int(round(max(height_left, height_right)))
    if out_w < 20 or out_h < 20:
        return board_crop, False, None, None, None

    # Tránh chấp nhận contour nhỏ/nhiễu hoặc tứ giác gần như suy biến.
    area_ratio = cv2.contourArea(quad.reshape(-1, 1, 2)) / float(w * h)
    if area_ratio < 0.15 or area_ratio > 0.98:
        return board_crop, False, None, None, None

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(
        board_crop, matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified, True, matrix, src, dst

def crop_one(image: np.ndarray, max_side: int, mask_background: bool = True, perspective_rectified: bool = True):
    small, scale = downscale(image, max_side)
    loc, feats = localize(small)
    x0, y0, x1, y1 = loc.bbox

    H, W = image.shape[:2]
    inv = 1.0 / scale
    X0 = max(0, int(math.floor(x0 * inv)))
    Y0 = max(0, int(math.floor(y0 * inv)))
    X1 = min(W, int(math.ceil(x1 * inv)))
    Y1 = min(H, int(math.ceil(y1 * inv)))
    crop = image[Y0:Y1, X0:X1].copy()

    mask_small = cleanup_foreground_mask(small, loc.bbox, feats) if mask_background else np.ones((y1 - y0, x1 - x0), np.uint8) * 255
    if mask_background:
        mask = cv2.resize(mask_small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        crop[mask == 0] = 255
    else:
        mask = None

    crop_shape_before_perspective = list(crop.shape[:2])
    perspective_matrix = None
    perspective_src_points = None
    perspective_dst_points = None
    rectified = False
    if perspective_rectified:
        crop, rectified, perspective_matrix, perspective_src_points, perspective_dst_points = rectify_perspective(crop)

    debug = {
        "small": small,
        "bbox_low": loc.bbox,
        "core": loc.debug_core,
        "method": loc.method,
        "confidence": loc.confidence,
        "mask_small": mask_small,
        "perspective_rectified": rectified,
        "crop_bbox_original": [int(X0), int(Y0), int(X1), int(Y1)],
        "crop_shape_before_perspective": crop_shape_before_perspective,
        "perspective_matrix": perspective_matrix.tolist() if perspective_matrix is not None else None,
        "perspective_src_points": perspective_src_points.tolist() if perspective_src_points is not None else None,
        "perspective_dst_points": perspective_dst_points.tolist() if perspective_dst_points is not None else None,
        "perspective_shape": list(crop.shape[:2]),
        "crop_method": loc.method,
        "crop_confidence": float(loc.confidence),
        "downscale_factor": float(scale),
        "bbox_low_resolution": [int(x0), int(y0), int(x1), int(y1)],
    }
    return crop, debug

def crop(image: np.ndarray, 
         max_side: int = 1500, 
         mask_background: bool = False,
         perspective_rectified: bool = True):
    
    cropped_image, debug_info = crop_one(image, 
                                max_side=max_side,
                                mask_background=mask_background,
                                perspective_rectified=perspective_rectified)

    # Ảnh debug ở đúng resolution dùng cho localization
    debug_img = debug_info["small"].copy()

    # Bbox trên ảnh small
    x0, y0, x1, y1 = debug_info["bbox_low"]

    # Vẽ bbox
    cv2.rectangle(debug_img, (x0, y0), (x1 - 1, y1 - 1),
        (0, 0, 255), 3,)

    # Thông tin phương pháp
    method = debug_info["method"]
    confidence = debug_info["confidence"]

    cv2.putText(debug_img, f"Method: {method}", (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
        2, cv2.LINE_AA,)

    cv2.putText(debug_img, f"Confidence: {confidence:.4f}",
        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 255, 0), 2, cv2.LINE_AA,)

    rectified_text = f"Perspective rectified: {debug_info['perspective_rectified']}"
    cv2.putText(debug_img, rectified_text, (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    return cropped_image, debug_img, debug_info