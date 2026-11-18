
from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# BASIC UTILITIES
# ============================================================

def odd(v: float, minimum: int = 3, maximum: int | None = None) -> int:
    """Tạo kích thước kernel lẻ trong giới hạn cho phép."""
    n = max(minimum, int(round(v)))
    if maximum is not None:
        n = min(n, maximum)
    if n % 2 == 0:
        n += 1
    if maximum is not None and n > maximum:
        n -= 2
    return max(3, n)


def percentile_stretch(
    gray: np.ndarray,
    lo: float = 1.0,
    hi: float = 99.0,
) -> np.ndarray:
    """Kéo giãn tương phản theo percentile."""
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError(f"Percentile không hợp lệ: lo={lo}, hi={hi}")

    a, b = np.percentile(gray, [lo, hi])
    if b <= a + 1e-6:
        return gray.copy()

    out = (gray.astype(np.float32) - a) * (255.0 / (b - a))
    return np.clip(out, 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh grayscale/BGR/BGRA về grayscale uint8."""
    if image is None:
        raise ValueError("Ảnh đầu vào là None.")

    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def clahe_gray(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    h, w = gray.shape
    grid_size = 12 if max(h, w) >= 3500 else 8
    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(grid_size, grid_size),
    )
    return clahe.apply(gray)


def unsharp(
    gray: np.ndarray,
    sigma: float = 1.0,
    amount: float = 0.50,
) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


# ============================================================
# GP: GAUSSIAN-BASED BACKGROUND ESTIMATION
# ============================================================

def apply_gp(
    image: np.ndarray,
    kernel_size_median_blur: int = 3,
    scale: float = 180.0,
    low_percentile_stretch_illum: float = 0.7,
    high_percentile_stretch_illum: float = 99.3,
) -> np.ndarray:
    """
    GP:
      1. Median Blur (giữ nguyên bước đang có trong code gốc).
      2. Gaussian-based Background Estimation.
      3. Illumination normalization.
      4. Percentile stretching.

    Median Blur được giữ trong GP để bảo toàn hành vi của pipeline cũ.
    """
    gray = to_gray(image)

    if kernel_size_median_blur and kernel_size_median_blur > 1:
        if kernel_size_median_blur % 2 == 0:
            raise ValueError("kernel_size_median_blur phải là số lẻ.")
        gray = cv2.medianBlur(gray, kernel_size_median_blur)

    h, w = gray.shape

    # Thu nhỏ ảnh khi cạnh lớn nhất vượt quá 1200 pixel.
    resize_scale = min(1.0, 1200.0 / max(h, w))
    if resize_scale < 1.0:
        small = cv2.resize(
            gray,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray

    # Gaussian-based Background Estimation.
    sigma = max(10.0, min(small.shape) * 0.022)
    background_small = cv2.GaussianBlur(
        small,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    # Khôi phục kích thước ảnh nền.
    if resize_scale < 1.0:
        background = cv2.resize(
            background_small,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        background = background_small

    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=scale)

    return percentile_stretch(
        normalized,
        low_percentile_stretch_illum,
        high_percentile_stretch_illum,
    )


# ============================================================
# CDE: CONTRAST AND DETAIL ENHANCEMENT
# ============================================================

def apply_cde(
    image: np.ndarray,
    minimum: int = 15,
    maximum: int = 71,
    low_percentile_stretch_tophat: float = 0.5,
    high_percentile_stretch_tophat: float = 99.5,
) -> np.ndarray:
    """
    CDE:
      1. CLAHE.
      2. Unsharp Masking.
      3. White Top-hat.
      4. Percentile stretching.
      5. Inversion để đưa nét chữ về màu tối.
    """
    gray = to_gray(image)

    # CLAHE và Unsharp Masking.
    enhanced = clahe_gray(gray, clip=2.0)
    enhanced = unsharp(enhanced, sigma=1.0, amount=0.50)

    h, w = enhanced.shape
    kernel_size = odd(
        min(h, w) * 0.013,
        minimum=minimum,
        maximum=maximum,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    # White Top-hat trích xuất các chi tiết sáng.
    bright_detail = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        kernel,
    )

    bright_detail = percentile_stretch(
        bright_detail,
        low_percentile_stretch_tophat,
        high_percentile_stretch_tophat,
    )

    # Đảo ảnh để nét chữ tối trên nền sáng.
    return 255 - bright_detail


# ============================================================
# MAIN PREPROCESSING / ABLATION
# ============================================================

VALID_MODES = {"full", "gp_only", "cde_only", "none"}


def flip_image(image: np.ndarray) -> np.ndarray:
    """Phép lật ngang cố định, áp dụng giống nhau ở mọi điều kiện."""
    return cv2.flip(image, 1)


def preprocess(
    image: np.ndarray,
    kernel_size_median_blur: int = 3,
    scale: float = 180.0,
    minimum: int = 15,
    maximum: int = 71,
    low_percentile_stretch_illum: float = 0.7,
    high_percentile_stretch_illum: float = 99.3,
    low_percentile_stretch_tophat: float = 0.5,
    high_percentile_stretch_tophat: float = 99.5,
    flip: bool = True,
    mode: str = "full",
) -> np.ndarray:
    """
    Các chế độ thí nghiệm:
      full     : GP -> CDE.
      gp_only  : chỉ thực hiện GP.
      cde_only : chỉ thực hiện CDE trên ảnh đầu vào, không chạy GP.
      none     : bỏ GP và CDE, giữ nguyên ảnh đầu vào.

    Phép lật ngang được giữ cố định ở cả bốn chế độ nếu flip=True.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode không hợp lệ: {mode!r}. "
            f"Chọn một trong {sorted(VALID_MODES)}."
        )

    if image is None:
        raise ValueError("Ảnh đầu vào là None.")

    if mode == "full":
        gp_result = apply_gp(
            image,
            kernel_size_median_blur=kernel_size_median_blur,
            scale=scale,
            low_percentile_stretch_illum=low_percentile_stretch_illum,
            high_percentile_stretch_illum=high_percentile_stretch_illum,
        )
        result = apply_cde(
            gp_result,
            minimum=minimum,
            maximum=maximum,
            low_percentile_stretch_tophat=low_percentile_stretch_tophat,
            high_percentile_stretch_tophat=high_percentile_stretch_tophat,
        )

    elif mode == "gp_only":
        result = apply_gp(
            image,
            kernel_size_median_blur=kernel_size_median_blur,
            scale=scale,
            low_percentile_stretch_illum=low_percentile_stretch_illum,
            high_percentile_stretch_illum=high_percentile_stretch_illum,
        )

    elif mode == "cde_only":
        # CDE được chạy trực tiếp trên ảnh đầu vào, không có GP/Median Blur.
        result = apply_cde(
            image,
            minimum=minimum,
            maximum=maximum,
            low_percentile_stretch_tophat=low_percentile_stretch_tophat,
            high_percentile_stretch_tophat=high_percentile_stretch_tophat,
        )

    else:  # mode == "none"
        # Giữ nguyên ảnh gốc, kể cả số kênh màu.
        result = image.copy()

    if flip:
        result = flip_image(result)

    return result
