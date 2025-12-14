from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
try:
    from scipy import ndimage as ndi  # type: ignore
except Exception:
    ndi = None  # type: ignore
from skimage.filters import median
from skimage.morphology import disk


def _import_kmeans():
    """惰性导入 KMean 若不可用则返回 None。"""
    try:
        from sklearn.cluster import KMeans  # type: ignore
        return KMeans
    except Exception:
        return None

class BaseFilter(ABC):

    @abstractmethod
    def apply(self, img: Image.Image) -> Image.Image:  # pragma: no cover
        raise NotImplementedError


class NegativeFilter(BaseFilter):

    def apply(self, img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")
        return ImageOps.invert(rgb)


class GrayFilter(BaseFilter):

    def apply(self, img: Image.Image) -> Image.Image:
        return img.convert("L")

class MedianFilter(BaseFilter):

    def apply(self, img: Image.Image, size: int = 3, roi: tuple[int, int, int, int] | None = None) -> Image.Image:
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]
        x0 = y0 = x1 = y1 = 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            x0 = max(0, min(w, int(x0)))
            x1 = max(0, min(w, int(x1)))
            y0 = max(0, min(h, int(y0)))
            y1 = max(0, min(h, int(y1)))
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 <= x0 or y1 <= y0:
                return Image.fromarray(arr.astype(np.uint8))

        if arr.ndim == 2:
            if ndi is not None:
                filtered = ndi.median_filter(arr, size=size, mode="reflect")
            else:
                fp = np.ones((size, size), dtype=bool)
                try:
                    filtered = median(arr, footprint=fp)
                except TypeError:
                    filtered = median(arr, selem=fp)
            if roi is None:
                out = filtered
            else:
                out = arr.copy()
                out[y0:y1, x0:x1] = filtered[y0:y1, x0:x1]
            return Image.fromarray(out.astype(np.uint8))
        elif arr.ndim == 3:
            if ndi is not None:
                filtered = ndi.median_filter(arr, size=(size, size, 1), mode="reflect")
            else:
                fp = np.ones((size, size), dtype=bool)
                try:
                    filtered = median(arr, footprint=fp, channel_axis=-1)
                except TypeError:
                    filtered = np.stack([median(arr[..., ch], fp) for ch in range(arr.shape[-1])], axis=-1)
            if roi is None:
                out = filtered
            else:
                out = arr.copy()
                out[y0:y1, x0:x1, :] = filtered[y0:y1, x0:x1, :]
            return Image.fromarray(out.astype(np.uint8))
        else:
            raise ValueError("Unsupported image ndim for median filter")

LAPLACIAN_4 = np.array(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]], dtype=np.float32
)

LAPLACIAN_8 = np.array(
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]], dtype=np.float32
)

class LaplacianEdgeFilter(BaseFilter):

    def __init__(self, kernel: np.ndarray | str = LAPLACIAN_4) -> None:
        if isinstance(kernel, str):
            kernel = LAPLACIAN_4 if kernel == "4" else LAPLACIAN_8
        self.kernel = kernel.astype(np.float32)

    def apply(self, img: Image.Image) -> Image.Image:
        if ndi is None:
            raise RuntimeError("scipy.ndimage is required for LaplacianEdgeFilter")
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)
        resp = ndi.convolve(arr, self.kernel, mode="reflect")
        edge = np.abs(resp)
        maxv = edge.max()
        if maxv > 0:
            edge = (edge / maxv) * 255.0
        edge = edge.astype(np.uint8)
        return Image.fromarray(edge)

class LaplacianSharpenFilter(BaseFilter):

    def __init__(self, kernel: np.ndarray | str = LAPLACIAN_4, amount: float = 1.0, strength: int | None = None) -> None:
        if isinstance(kernel, str):
            kernel = LAPLACIAN_4 if kernel == "4" else LAPLACIAN_8
        self.kernel = kernel.astype(np.float32)
        if strength is not None:
            s = max(0, min(int(strength), 100))
            self.amount = s / 100.0
        else:
            self.amount = float(amount)

    def apply(self, img: Image.Image) -> Image.Image:
        if ndi is None:
            raise RuntimeError("scipy.ndimage is required for LaplacianSharpenFilter")
        rgb = img.convert("RGB")
        arr_rgb = np.array(rgb, dtype=np.float32)
        gray = rgb.convert("L")
        arr = np.array(gray, dtype=np.float32)
        resp = ndi.convolve(arr, self.kernel, mode="reflect")
        edge = np.abs(resp)
        maxv = edge.max()
        if maxv > 0:
            edge = (edge / maxv) * 255.0
        e3 = np.stack([edge, edge, edge], axis=-1)
        out = arr_rgb + self.amount * e3
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

class PaletteQuantizer(BaseFilter):
    """
    调色盘量化滤镜( KMeans 实现）。
    将图像像素展平后使用 KMeans 聚类，
    以聚类中心作为调色盘颜色重建图像。
    """

    def __init__(self, n_colors: int = 2, random_state: Optional[int] = 0, fallback: bool = False) -> None:
        self.n_colors = max(1, min(int(n_colors), 256))
        self.random_state = random_state
        self.fallback = fallback

    def apply(self, img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")

        KMeans = _import_kmeans()
        if KMeans is not None and not self.fallback:
            # 使用 KMeans 聚类
            w, h = rgb.size
            arr = np.array(rgb, dtype=np.uint8).reshape(-1, 3)
            kmeans = KMeans(n_clusters=self.n_colors, random_state=self.random_state)
            kmeans.fit(arr)
            palette = kmeans.cluster_centers_.astype(np.uint8)
            labels = kmeans.labels_
            new_arr = palette[labels].reshape(h, w, 3)
            return Image.fromarray(new_arr, mode="RGB")
        else:
            # 备用：使用 PIL 的 quantize（不完全等价于 KMeans，但效果接近）
            pal = rgb.quantize(colors=self.n_colors, method=1, dither=Image.NONE)
            return pal.convert("RGB")


__all__ = [
    "BaseFilter",
    "NegativeFilter",
    "GrayFilter",
    "PaletteQuantizer",
    "MedianFilter",
    "LaplacianEdgeFilter",
    "LaplacianSharpenFilter",
]


if __name__ == "__main__":  # 简单示例
    # 示例：读取图片并应用三种滤镜
    try:
        img = Image.open("hw1/experiment.png").convert("RGB")
    except Exception as e:
        raise RuntimeError("Failed to load image.") from e

    # neg = NegativeFilter().apply(img)
    # gray = GrayFilter().apply(img)
    # pal = PaletteQuantizer(n_colors=8).apply(img)
    med = MedianFilter().apply(img, size=3)
    lap = LaplacianEdgeFilter(kernel="4").apply(img)
    sharp = LaplacianSharpenFilter(kernel="4", amount=1.0).apply(img)

    # neg.save("hw3/_demo_negative.png")
    # gray.save("hw3/_demo_gray.png")
    # pal.save("hw3/_demo_palette.png")
    med.save("hw3/_demo_median.png")
    lap.save("hw3/_demo_laplacian.png")
    sharp.save("hw3/_demo_sharpen.png")