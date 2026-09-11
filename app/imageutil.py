# -*- coding: utf-8 -*-
"""图像导入、灰度/透光率映射、缩略图与操作单图像导出。"""
import io

import numpy as np
from PIL import Image, ImageDraw

MAX_PREVIEW = 1600
THUMB_W = 240
RESULT_W = 640


def load_image(path):
    img = Image.open(path)
    img.load()
    return img


def import_image(src_path, out_path, max_side=MAX_PREVIEW):
    """保存原图（转 PNG，保留原色信息）并返回 (宽, 高, 预览 numpy 灰度)。"""
    img = load_image(src_path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    img.save(out_path)
    return _preview_gray(img)


def _preview_gray(img):
    im = img
    if max(im.size) > MAX_PREVIEW:
        ratio = MAX_PREVIEW / max(im.size)
        im = im.resize((round(im.width * ratio), round(im.height * ratio)),
                       Image.LANCZOS)
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    return im.width, im.height, gray


def base_map_from_file(path, invert, max_side=MAX_PREVIEW):
    """读取已保存的原图，返回 (W,H,gray0)。
    底片扫描默认把"胶片黑"当作透光少：T = 1 - L/255；
    invert=1 时按正像直接使用亮度。"""
    img = load_image(path)
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((round(img.width * ratio), round(img.height * ratio)),
                         Image.LANCZOS)
    lum = np.asarray(img.convert("L"), dtype=np.float32)
    T = (255.0 - lum) if not invert else lum
    # 纸基灰底 245，最深 12，避免 0/255 造成对数发散
    gray0 = 245.0 - T / 255.0 * (245.0 - 12.0)
    return img.width, img.height, gray0.astype(np.float32)


def encode_png(arr_gray):
    a = np.clip(arr_gray, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, "L").save(buf, "PNG")
    return buf.getvalue()


def result_png(gray, base_gray, masks=None, overlays=None, width=RESULT_W):
    """最终灰度预览 PNG（可叠加彩色工具覆盖）。"""
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), "L").convert("RGB")
    ratio = width / img.width
    if ratio != 1.0:
        img = img.resize((width, round(img.height * ratio)), Image.LANCZOS)

    if overlays:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        W0, H0 = gray.shape[1], gray.shape[0]
        for i, cov in enumerate(overlays):
            color = PALETTE[i % len(PALETTE)]
            alpha = (np.clip(cov, 0, 1) * 90).astype(np.uint8)
            am = Image.fromarray(alpha, "L")
            if am.size != img.size:
                am = am.resize(img.size, Image.BILINEAR)
            solid = Image.new("RGBA", img.size, color + (0,))
            solid.putalpha(am)
            layer = Image.alpha_composite(layer, solid)
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def mask_thumb(cov, label, width=THUMB_W):
    """单个工具遮罩缩略图（黑底橙膜 + 名称），返回 PNG bytes。"""
    H, W = cov.shape
    h = max(40, round(H / W * width))
    am = Image.fromarray((np.clip(cov, 0, 1) * 255).astype(np.uint8), "L")
    am = am.resize((width, h), Image.BILINEAR)
    img = Image.new("RGB", (width, h), (18, 18, 18))
    film = Image.new("RGBA", (width, h), (255, 140, 0, 0))
    film.putalpha(am.point(lambda v: int(v * 0.75)))
    img = Image.alpha_composite(img.convert("RGBA"), film).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, h - 16, width, h], fill=(0, 0, 0))
    d.text((4, h - 14), label[:28], fill=(255, 200, 120))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


PALETTE = [
    (255, 140, 0),    # 加光-橙
    (80, 170, 255),   # 遮挡-蓝
    (180, 255, 80),
    (255, 90, 200),
    (180, 120, 255),
    (60, 220, 200),
]
