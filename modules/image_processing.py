"""
SAED Image Processing & Center Detection Module
Provides routines for image preprocessing, contrast enhancement, cropping,
and direct-beam / pattern center auto-detection (Hough circles, intensity centroid, MinMax/Blob).
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2
from scipy import ndimage


def load_image_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """Load an image from raw bytes into a float grayscale image [0, 1]."""
    # Check if dm3 / dm4
    lower = filename.lower()
    if lower.endswith((".dm3", ".dm4")):
        try:
            from ncempy.io import dm
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=lower[lower.rfind("."):], delete=True) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                data = dm.dmReader(tmp.name)
                img = np.squeeze(data.get("data", data))
                img = img.astype(np.float32)
                img = (img - np.min(img)) / (np.ptp(img) + 1e-8)
                return img
        except Exception:
            pass

    # Standard image formats (PNG, JPG, TIFF, etc.)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {filename}")

    if img_bgr.ndim == 3:
        if img_bgr.shape[2] == 4:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    img_float = img_gray.astype(np.float32)
    min_v, max_v = np.min(img_float), np.max(img_float)
    if max_v > min_v:
        img_float = (img_float - min_v) / (max_v - min_v)
    else:
        img_float = np.zeros_like(img_float)

    return img_float


def adjust_contrast(
    image: np.ndarray,
    contrast_low: float = 0.5,
    contrast_high: float = 99.5,
    gamma: float = 1.0,
    invert: bool = False
) -> np.ndarray:
    """Enhance contrast via percentile stretching and gamma correction."""
    img = image.copy()
    if invert:
        img = 1.0 - img

    p_low, p_high = np.percentile(img, (contrast_low, contrast_high))
    if p_high > p_low:
        img = np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img = np.clip(img, 0.0, 1.0)

    if gamma != 1.0 and gamma > 0:
        img = np.power(img, 1.0 / gamma)

    return np.clip(img, 0.0, 1.0)


def crop_image(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int
) -> np.ndarray:
    """Crop image to [top:bottom, left:right] with boundary clamping."""
    h, w = image.shape[:2]
    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))
    return image[top:bottom, left:right]


def detect_center_centroid(
    image: np.ndarray,
    threshold_percentile: float = 98.0,
    sigma: float = 2.0
) -> Tuple[float, float]:
    """Detect beam center using weighted center of mass of brightest saturated region."""
    smoothed = ndimage.gaussian_filter(image, sigma=sigma)
    thresh = np.percentile(smoothed, threshold_percentile)
    mask = smoothed >= thresh

    if not np.any(mask):
        h, w = image.shape
        return float(w / 2.0), float(h / 2.0)

    # Weighted center of mass: (cy, cx) in scipy coordinates
    cy, cx = ndimage.center_of_mass(smoothed * mask)
    if np.isnan(cx) or np.isnan(cy):
        h, w = image.shape
        return float(w / 2.0), float(h / 2.0)

    return float(cx), float(cy)


def detect_center_hough(
    image: np.ndarray,
    min_radius: int = 15,
    max_radius: int = 120,
    dp: float = 1.2,
    param1: float = 50.0,
    param2: float = 30.0
) -> Optional[Tuple[float, float, float]]:
    """Detect central beam or concentric ring circle using circular Hough transform."""
    img_u8 = (image * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_radius * 2,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None and len(circles) > 0:
        c = circles[0]
        # Sort circles by proximity to geometric center
        h, w = image.shape
        cx_img, cy_img = w / 2.0, h / 2.0
        sorted_circles = sorted(c, key=lambda circ: (circ[0] - cx_img)**2 + (circ[1] - cy_img)**2)
        best = sorted_circles[0]
        return float(best[0]), float(best[1]), float(best[2])

    return None


def detect_center_minmax(image: np.ndarray, sigma: float = 3.0) -> Tuple[float, float]:
    """Detect center by locating the maximum intensity location after smoothing."""
    smoothed = ndimage.gaussian_filter(image, sigma=sigma)
    idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    cy, cx = idx[0], idx[1]
    return float(cx), float(cy)


def auto_detect_center(
    image: np.ndarray,
    method: str = "Intensity Centroid",
    **kwargs: Any
) -> Tuple[float, float, Dict[str, Any]]:
    """Universal dispatcher for auto-detecting beam center."""
    details: Dict[str, Any] = {"method": method}
    h, w = image.shape[:2]

    if method == "Circular Hough Transform":
        res = detect_center_hough(
            image,
            min_radius=kwargs.get("min_radius", 20),
            max_radius=kwargs.get("max_radius", min(h, w) // 4),
            param1=kwargs.get("param1", 50.0),
            param2=kwargs.get("param2", 30.0)
        )
        if res is not None:
            details["detected_radius"] = res[2]
            return res[0], res[1], details
        # Fallback to centroid if Hough finds no circle
        details["hough_fallback"] = True
        cx, cy = detect_center_centroid(image, threshold_percentile=kwargs.get("threshold_percentile", 98.0))
        return cx, cy, details

    elif method == "Max Intensity Peak":
        cx, cy = detect_center_minmax(image, sigma=kwargs.get("sigma", 3.0))
        return cx, cy, details

    else:  # Default to Intensity Centroid
        thresh = kwargs.get("threshold_percentile", 98.0)
        sigma = kwargs.get("sigma", 2.0)
        cx, cy = detect_center_centroid(image, threshold_percentile=thresh, sigma=sigma)
        return cx, cy, details


def detect_2d_diffraction_spots(
    image: np.ndarray,
    center: Tuple[float, float],
    min_radius_px: float = 15.0,
    max_radius_px: Optional[float] = None,
    threshold_rel: float = 0.08,
    min_distance_px: int = 5
) -> List[Dict[str, float]]:
    """Detect discrete 2D diffraction spots in textured/single-crystal patterns using local maxima."""
    h, w = image.shape[:2]
    cx, cy = center
    if max_radius_px is None:
        max_radius_px = float(min(cx, cy, w - cx, h - cy))

    # Mild smoothing to eliminate single dead/hot pixel noise
    smoothed = ndimage.gaussian_filter(image, sigma=1.0)

    # Local maximum filter
    neighborhood_size = max(3, int(min_distance_px * 2 + 1))
    local_max = ndimage.maximum_filter(smoothed, size=neighborhood_size) == smoothed

    # Thresholding
    thresh = float(np.min(smoothed) + threshold_rel * np.ptp(smoothed))
    detected = local_max & (smoothed >= thresh)

    y_coords, x_coords = np.where(detected)
    spots: List[Dict[str, float]] = []

    for y, x in zip(y_coords, x_coords):
        dx = float(x - cx)
        dy = float(y - cy)
        r = float(np.hypot(dx, dy))

        if min_radius_px <= r <= max_radius_px:
            val = float(image[y, x])
            theta = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
            spots.append({
                "x": float(x),
                "y": float(y),
                "radius_px": round(r, 2),
                "intensity": round(val, 4),
                "angle_deg": round(theta, 1)
            })

    # Sort by distance from center
    spots.sort(key=lambda s: s["radius_px"])
    return spots

