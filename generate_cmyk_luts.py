# -*- coding: utf-8 -*-
"""生成模拟「RGB 图像送去印刷、转成 CMYK」时几种典型现象的 3D LUT（.cube）。

这些 LUT 不是某一台打印机的 ICC 特性文件，而是在 OKLCH 里用一个
「铜版纸胶印」形状的色域做的可视近似，方便在 dwm_lut 里直接套到屏幕上对比：

  CMYK-1-色域截断.cube      超色域的颜色被切到色域边界（色相保留，亮的艳色会变暗）
  CMYK-2-拉向相邻色.cube    超色域的颜色除了被截断，色相还会被拽向打印机真正印得出来的邻近色
  CMYK-3-亮紫警告.cube      超色域的颜色整块换成亮紫色（类似 Photoshop 色域警告）
  CMYK-4-整体压缩.cube      整个 RGB 色域被平滑挤进 CMYK，色域内的颜色也会变灰一点
  CMYK-5-综合印刷.cube      压缩 + 轻微色相偏移 + 印刷黑不够黑 + 纸白偏暖

用法：
  python generate_cmyk_luts.py
  python generate_cmyk_luts.py --size 33
  python generate_cmyk_luts.py --preview preview.png

.cube 为 Adobe / DisplayCAL 顺序：蓝最慢、绿居中、红最快。dwm_lut 按这个顺序读取。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

LUT_SIZE = 33
OUT_DIR = Path(__file__).resolve().parent

# 亮紫警告色。要够亮、够紫，套在桌面上能一眼分开。
WARNING_RGB = (0.80, 0.00, 1.00)

# 色域肩部。1 是尖三角形；越小，中等明度能保住的彩度越多。
GAMUT_FALLOFF = 0.62

# (色相度数, 该色相彩度最大时的明度, 最大彩度)
# 黄、红比较能印；绿和蓝明显收；青是印刷的强项。
CUSPS = (
    (0.0, 0.58, 0.170),
    (30.0, 0.62, 0.190),
    (55.0, 0.70, 0.180),
    (85.0, 0.82, 0.170),
    (110.0, 0.91, 0.160),
    (130.0, 0.74, 0.130),
    (150.0, 0.66, 0.115),
    (175.0, 0.68, 0.125),
    (200.0, 0.75, 0.140),
    (225.0, 0.62, 0.120),
    (255.0, 0.52, 0.110),
    (280.0, 0.50, 0.115),
    (305.0, 0.54, 0.140),
    (330.0, 0.58, 0.160),
    (360.0, 0.58, 0.170),
)

# 超色域时把色相拽向「邻近的可印色」。
# (区域中心色相, 目标色相, 宽度, 最多走完这段距离的比例)
# 绿往黄，蓝往紫，品红往红，电光青略往印刷青。
HUE_TARGETS = (
    (142.0, 114.0, 26.0, 0.80),
    (264.0, 308.0, 32.0, 0.90),
    (328.0, 356.0, 26.0, 0.72),
    (196.0, 210.0, 18.0, 0.50),
)

# 彩度低于这个值视为中性色，印刷也能还原，不做色域处理。
NEUTRAL_C = 0.004


def cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def srgb_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    r, g, b = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, a, b_


def oklab_to_linear(L: float, a: float, b: float) -> tuple[float, float, float]:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_ = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return r, g, b_


def oklab_to_srgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    """转回 sRGB。若落在立方体外，就降低彩度直到进去，避免硬切通道把色相带偏。"""
    for _ in range(16):
        lr, lg, lb = oklab_to_linear(L, a, b)
        if (
            -1e-4 <= lr <= 1.0 + 1e-4
            and -1e-4 <= lg <= 1.0 + 1e-4
            and -1e-4 <= lb <= 1.0 + 1e-4
        ):
            return (
                linear_to_srgb(min(max(lr, 0.0), 1.0)),
                linear_to_srgb(min(max(lg, 0.0), 1.0)),
                linear_to_srgb(min(max(lb, 0.0), 1.0)),
            )
        a *= 0.86
        b *= 0.86
    lr, lg, lb = oklab_to_linear(L, a, b)
    return (
        linear_to_srgb(min(max(lr, 0.0), 1.0)),
        linear_to_srgb(min(max(lg, 0.0), 1.0)),
        linear_to_srgb(min(max(lb, 0.0), 1.0)),
    )


def lab_to_lch(a: float, b: float) -> tuple[float, float]:
    return math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360.0


def lch_to_lab(c: float, h: float) -> tuple[float, float]:
    rad = math.radians(h)
    return c * math.cos(rad), c * math.sin(rad)


def ang_from_to(src: float, dst: float) -> float:
    """从 src 转到 dst 的有符号角度，范围 (-180, 180]。"""
    return (dst - src + 180.0) % 360.0 - 180.0


def cusp(h: float) -> tuple[float, float]:
    h = h % 360.0
    for i in range(len(CUSPS) - 1):
        h0, l0, c0 = CUSPS[i]
        h1, l1, c1 = CUSPS[i + 1]
        if h0 <= h <= h1:
            t = 0.0 if h1 <= h0 else (h - h0) / (h1 - h0)
            return l0 + (l1 - l0) * t, c0 + (c1 - c0) * t
    return CUSPS[0][1], CUSPS[0][2]


def cmax(h: float, L: float) -> float:
    """这个色相、这个明度下，模型允许的最大彩度。"""
    L = min(max(L, 0.0), 1.0)
    if L <= 0.0 or L >= 1.0:
        return 0.0
    lc, cc = cusp(h)
    if lc <= 1e-8 or lc >= 1.0:
        return 0.0
    t = (L / lc) if L <= lc else ((1.0 - L) / (1.0 - lc))
    return cc * (t ** GAMUT_FALLOFF)


def is_neutral(r: float, g: float, b: float) -> bool:
    return abs(r - g) < 1e-7 and abs(g - b) < 1e-7


def outside_gamut(L: float, c: float, h: float) -> bool:
    if c <= NEUTRAL_C:
        return False
    return c > cmax(h, L) + 0.0015


def clip_oklch(L: float, c: float, h: float) -> tuple[float, float]:
    """截断到色域内。

    比打印机更亮、又太艳的颜色，会往该色相的「最艳可印点」压暗再砍彩度
    （油墨加上去，亮艳色会变暗）。已经比这个点更暗的颜色只砍彩度，阴影保持暗。
    """
    limit = cmax(h, L)
    if c <= limit:
        return L, c
    lc, cc = cusp(h)
    if L > lc and cc > 1e-8:
        excess = (c - limit) / cc
        pull = 1.0 - math.exp(-1.35 * excess)
        L2 = L + (lc - L) * pull
        return L2, min(c, cmax(h, L2))
    return L, limit


def hue_pull_degrees(h: float, excess: float) -> float:
    """excess 为 0..1，表示这个颜色有多超出色域。色域内应为 0。"""
    amount = min(max(excess, 0.0), 1.0) ** 0.75
    if amount <= 0.0:
        return 0.0
    shift = 0.0
    for center, target, width, frac in HUE_TARGETS:
        distance = ang_from_to(center, h)
        weight = math.exp(-0.5 * (distance / width) ** 2)
        shift += weight * frac * ang_from_to(h, target)
    return shift * amount


# sRGB 在固定明度、色相下的最大彩度。压缩模式要用它当「源色域」的边缘。
_SRC_L_N = 65
_SRC_H_N = 180
_src_chroma: list[float] | None = None


def _build_src_chroma() -> list[float]:
    table = [0.0] * (_SRC_L_N * _SRC_H_N)
    for il in range(_SRC_L_N):
        L = il / (_SRC_L_N - 1)
        for ih in range(_SRC_H_N):
            h = ih * (360.0 / _SRC_H_N)
            lo, hi = 0.0, 0.45
            for _ in range(16):
                mid = (lo + hi) * 0.5
                a, b = lch_to_lab(mid, h)
                lr, lg, lb = oklab_to_linear(L, a, b)
                inside = (
                    -2e-4 <= lr <= 1.0 + 2e-4
                    and -2e-4 <= lg <= 1.0 + 2e-4
                    and -2e-4 <= lb <= 1.0 + 2e-4
                )
                if inside:
                    lo = mid
                else:
                    hi = mid
            table[il * _SRC_H_N + ih] = lo
    return table


def srgb_max_chroma(L: float, h: float) -> float:
    global _src_chroma
    if _src_chroma is None:
        _src_chroma = _build_src_chroma()
    L = min(max(L, 0.0), 1.0)
    h = h % 360.0
    x_l = L * (_SRC_L_N - 1)
    x_h = h * (_SRC_H_N / 360.0)
    il = min(int(x_l), _SRC_L_N - 2)
    ih = int(x_h) % _SRC_H_N
    tl = x_l - il
    th = x_h - math.floor(x_h)
    ih2 = (ih + 1) % _SRC_H_N

    def at(l_i: int, h_i: int) -> float:
        return _src_chroma[l_i * _SRC_H_N + h_i]

    c00 = at(il, ih)
    c10 = at(il + 1, ih)
    c01 = at(il, ih2)
    c11 = at(il + 1, ih2)
    return (
        c00 * (1 - tl) * (1 - th)
        + c10 * tl * (1 - th)
        + c01 * (1 - tl) * th
        + c11 * tl * th
    )


def compress_oklch(L: float, c: float, h: float) -> tuple[float, float]:
    """把源色域平滑挤进印刷色域。低彩度少动，越靠近 sRGB 边缘压得越狠。

    即使某个色相上印刷色域和屏幕差不多宽，饱和色也要收一截。
    感知意图会给超色域的颜色腾位置，所以色域内部也不会原样保留。
    """
    c_src = srgb_max_chroma(L, h)
    if c <= NEUTRAL_C or c_src <= 1e-6:
        return L, 0.0 if c <= NEUTRAL_C else min(c, cmax(h, L))
    fullness = c / c_src
    c_dst = cmax(h, L)
    ratio = min(c_dst / c_src, 1.0)
    # 屏幕色域边缘最多保留 75%，或者更低（如果这个色相印刷色域更窄）。
    scale_at_rim = min(0.75, ratio * 0.96)
    w = min(fullness, 1.0) ** 0.55
    factor = 0.97 * (1.0 - w) + scale_at_rim * w
    if fullness > 1.0:
        factor *= 1.0 / fullness
    c2 = c * factor
    l_clip, _ = clip_oklch(L, c, h)
    L2 = L + (l_clip - L) * (min(fullness, 1.0) ** 0.85)
    c2 = min(max(c2, 0.0), cmax(h, L2) * 0.995)
    return L2, c2


def _rgb_to_lch(r: float, g: float, b: float) -> tuple[float, float, float]:
    L, a, b_ = srgb_to_oklab(r, g, b)
    c, h = lab_to_lch(a, b_)
    return L, c, h


def map_clip(r: float, g: float, b: float) -> tuple[float, float, float]:
    if is_neutral(r, g, b):
        return r, g, b
    L, c, h = _rgb_to_lch(r, g, b)
    if not outside_gamut(L, c, h):
        return r, g, b
    L2, c2 = clip_oklch(L, c, h)
    a, b_ = lch_to_lab(c2, h)
    return oklab_to_srgb(L2, a, b_)


def map_adjacent(r: float, g: float, b: float) -> tuple[float, float, float]:
    if is_neutral(r, g, b):
        return r, g, b
    L, c, h = _rgb_to_lch(r, g, b)
    if not outside_gamut(L, c, h):
        return r, g, b
    limit = cmax(h, L)
    excess = (c - limit) / max(c, 1e-6)
    h2 = (h + hue_pull_degrees(h, excess)) % 360.0
    L2, c2 = clip_oklch(L, c, h2)
    a, b_ = lch_to_lab(c2, h2)
    return oklab_to_srgb(L2, a, b_)


def map_warning(r: float, g: float, b: float) -> tuple[float, float, float]:
    if is_neutral(r, g, b):
        return r, g, b
    L, c, h = _rgb_to_lch(r, g, b)
    if outside_gamut(L, c, h):
        return WARNING_RGB
    return r, g, b


def map_compress(r: float, g: float, b: float) -> tuple[float, float, float]:
    if is_neutral(r, g, b):
        return r, g, b
    L, c, h = _rgb_to_lch(r, g, b)
    L2, c2 = compress_oklch(L, c, h)
    a, b_ = lch_to_lab(c2, h)
    return oklab_to_srgb(L2, a, b_)


def map_print(r: float, g: float, b: float) -> tuple[float, float, float]:
    """综合：先压缩，再把超色域色相轻轻拽向邻近可印色，最后加上纸白和印刷黑。"""
    L, a, b_ = srgb_to_oklab(r, g, b)
    c, h = lab_to_lch(a, b_)
    if c > NEUTRAL_C:
        limit = cmax(h, L)
        if c > limit:
            excess = (c - limit) / max(c, 1e-6)
            h = (h + 0.55 * hue_pull_degrees(h, excess)) % 360.0
        L, c = compress_oklch(L, c, h)
        a, b_ = lch_to_lab(c, h)
    # 印刷黑抬不上屏幕黑，只抬最深的阴影；纸白略降并且偏暖。
    lift = 0.175 * math.exp(-((L / 0.16) ** 2))
    L = L * 0.965 + lift
    a += 0.005 * L
    b_ += 0.014 * L
    return oklab_to_srgb(L, a, b_)


MODES = (
    {
        "filename": "CMYK-1-色域截断.cube",
        "title": "CMYK clip",
        "comment": "超色域颜色被截断到印刷色域边界。色相保留；亮而艳的颜色会变暗、变灰。色域内的颜色原样通过。",
        "fn": map_clip,
    },
    {
        "filename": "CMYK-2-拉向相邻色.cube",
        "title": "CMYK adjacent hue",
        "comment": "超色域颜色会被拉向邻近的可印色：绿偏黄、蓝偏紫、品红偏红，然后再截断。色域内的颜色原样通过。",
        "fn": map_adjacent,
    },
    {
        "filename": "CMYK-3-亮紫警告.cube",
        "title": "CMYK gamut warning",
        "comment": "超出印刷色域的颜色替换成亮紫色，作为色域警告。印得出来的颜色保持原样。",
        "fn": map_warning,
    },
    {
        "filename": "CMYK-4-整体压缩.cube",
        "title": "CMYK perceptual compress",
        "comment": "整个 RGB 色域被平滑压进 CMYK。不只是最艳的颜色，中等饱和的颜色也会收一点。没有硬边。",
        "fn": map_compress,
    },
    {
        "filename": "CMYK-5-综合印刷.cube",
        "title": "CMYK print look",
        "comment": "压缩进色域，并带一点邻近色偏移、印刷黑抬升和暖纸白。接近软打样的综合观感。",
        "fn": map_print,
    },
)


def write_cube(path: Path, title: str, comment: str, size: int, fn) -> None:
    step = 1.0 / (size - 1)
    # 中文说明放在 LUT_3D_SIZE 之前。dwm_lut 找到尺寸后只接受数字行，
    # 之后不能再写非 ASCII，否则 signed char 比较会把 UTF-8 误当成数据。
    header = [
        f'TITLE "{title}"',
        f"# {comment}",
        "# RGB to CMYK print-gamut approximation. Not a measured ICC profile.",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        f"LUT_3D_SIZE {size}",
    ]
    data: list[str] = []
    for ib in range(size):
        b = ib * step
        for ig in range(size):
            g = ig * step
            for ir in range(size):
                r = ir * step
                ro, go, bo = fn(r, g, b)
                data.append(f"{ro:.6f} {go:.6f} {bo:.6f}")
    text = "\n".join(header) + "\n" + "\n".join(data) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _node_index(size: int, r: float, g: float, b: float) -> int:
    step = 1.0 / (size - 1)
    ir = int(round(r / step))
    ig = int(round(g / step))
    ib = int(round(b / step))
    return (ib * size + ig) * size + ir


def _read_nodes(path: Path) -> tuple[int, list[tuple[float, float, float]]]:
    size = None
    nodes: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("LUT_3D_SIZE"):
            size = int(s.split()[-1])
            continue
        if not s or s.startswith("#") or s.startswith("TITLE") or s.startswith("DOMAIN"):
            continue
        parts = s.split()
        if len(parts) == 3:
            nodes.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if size is None:
        raise RuntimeError(f"{path.name} 缺少 LUT_3D_SIZE")
    return size, nodes


def _close(a: tuple[float, float, float], b: tuple[float, float, float], tol: float = 2e-3) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol and abs(a[2] - b[2]) < tol


def self_check(paths: list[Path], size: int) -> None:
    skin = (0.91, 0.75, 0.65)
    teal = (0.0, 0.50, 0.50)
    gray = (0.5, 0.5, 0.5)

    assert _close(map_clip(*gray), gray, 1e-6)
    assert _close(map_warning(*gray), gray, 1e-6)
    assert _close(map_compress(*gray), gray, 1e-6)
    assert _close(map_clip(1, 1, 1), (1, 1, 1), 1e-6)
    assert _close(map_clip(0, 0, 0), (0, 0, 0), 1e-6)
    assert _close(map_warning(*skin), skin, 1e-6)

    for src in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1)):
        assert _close(map_warning(*src), WARNING_RGB, 1e-6), src

    def hue_of(rgb: tuple[float, float, float]) -> float:
        return _rgb_to_lch(*rgb)[2]

    blue_clip_h = hue_of(map_clip(0, 0, 1))
    blue_adj_h = hue_of(map_adjacent(0, 0, 1))
    green_clip_h = hue_of(map_clip(0, 1, 0))
    green_adj_h = hue_of(map_adjacent(0, 1, 0))
    # 蓝被拉向紫（色相增大），绿被拉向黄（色相减小）。
    assert ang_from_to(blue_clip_h, blue_adj_h) > 8.0, (blue_clip_h, blue_adj_h)
    assert ang_from_to(green_adj_h, green_clip_h) > 8.0, (green_clip_h, green_adj_h)

    teal_c = _rgb_to_lch(*teal)[1]
    teal_compressed_c = _rgb_to_lch(*map_compress(*teal))[1]
    assert teal_compressed_c < teal_c * 0.92, (teal_c, teal_compressed_c)
    # 青属于色域内，截断模式应保持原色，这样才看得出「压缩」和「截断」的差别。
    assert _close(map_clip(*teal), teal, 2e-3)

    pw = map_print(1, 1, 1)
    assert pw[0] < 0.99 and pw[1] < 0.99 and pw[2] < pw[0] - 0.01, pw
    pb = map_print(0, 0, 0)
    assert min(pb) > 0.04, pb

    samples = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0),
        skin, teal, gray, (1, 1, 1), (0, 0, 0), (1, 0.5, 0), (0.2, 0.55, 0.9),
    )
    for fn in (map_clip, map_adjacent, map_warning, map_compress, map_print):
        for src in samples:
            out = fn(*src)
            assert all(0.0 <= x <= 1.0 for x in out), (fn.__name__, src, out)

    for path, mode in zip(paths, MODES):
        got_size, nodes = _read_nodes(path)
        assert got_size == size
        assert len(nodes) == size ** 3, (path.name, len(nodes))
        for src in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5), (1, 1, 1)):
            node = nodes[_node_index(size, *src)]
            expect = mode["fn"](*src)
            assert _close(node, expect, 1.5e-3), (path.name, src, node, expect)


def diagnose() -> None:
    swatches = (
        ("红", (1, 0, 0)),
        ("绿", (0, 1, 0)),
        ("蓝", (0, 0, 1)),
        ("青", (0, 1, 1)),
        ("品红", (1, 0, 1)),
        ("黄", (1, 1, 0)),
        ("肤色", (0.91, 0.75, 0.65)),
        ("青绿", (0, 0.5, 0.5)),
        ("天蓝", (0.6, 0.75, 0.95)),
        ("橙", (1, 0.5, 0)),
        ("海军蓝", (0, 0, 0.4)),
        ("灰", (0.5, 0.5, 0.5)),
        ("白", (1, 1, 1)),
        ("黑", (0, 0, 0)),
    )
    names = ("截断", "相邻", "警告", "压缩", "印刷")
    fns = (map_clip, map_adjacent, map_warning, map_compress, map_print)
    print("\n输入 RGB255  →  各 LUT 输出")
    for name, src in swatches:
        src8 = tuple(int(round(c * 255)) for c in src)
        cells = []
        for fn in fns:
            out = fn(*src)
            cells.append("({:3d},{:3d},{:3d})".format(*(int(round(c * 255)) for c in out)))
        print(f"{name:<4} {src8!s:<18} " + "  ".join(f"{n}:{c}" for n, c in zip(names, cells)))


def write_preview(path: Path) -> None:
    from PIL import Image, ImageDraw

    swatches = (
        ("R", (255, 0, 0)),
        ("G", (0, 255, 0)),
        ("B", (0, 0, 255)),
        ("C", (0, 255, 255)),
        ("M", (255, 0, 255)),
        ("Y", (255, 255, 0)),
        ("skin", (232, 191, 166)),
        ("teal", (0, 128, 128)),
        ("sky", (153, 191, 242)),
        ("orange", (255, 127, 0)),
        ("navy", (0, 0, 102)),
        ("leaf", (40, 140, 50)),
        ("gray", (128, 128, 128)),
        ("white", (255, 255, 255)),
        ("black", (0, 0, 0)),
    )
    cols = (
        ("source", None),
        ("clip", map_clip),
        ("adjacent", map_adjacent),
        ("warning", map_warning),
        ("compress", map_compress),
        ("print", map_print),
    )
    pad, label_w, cell, head = 8, 64, 72, 22
    width = pad + label_w + cell * len(cols) + pad
    height = pad + head + cell * len(swatches) + pad
    image = Image.new("RGB", (width, height), (32, 32, 32))
    draw = ImageDraw.Draw(image)
    for ci, (title, _) in enumerate(cols):
        draw.text((pad + label_w + ci * cell + 4, pad), title, fill=(230, 230, 230))
    for ri, (name, rgb) in enumerate(swatches):
        y = pad + head + ri * cell
        draw.text((pad, y + cell // 2 - 6), name, fill=(220, 220, 220))
        for ci, (_, fn) in enumerate(cols):
            x = pad + label_w + ci * cell
            if fn is None:
                color = rgb
            else:
                out = fn(*(c / 255.0 for c in rgb))
                color = tuple(int(round(c * 255)) for c in out)
            draw.rectangle((x + 4, y + 4, x + cell - 4, y + cell - 4), fill=color)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成模拟 RGB→CMYK 印刷现象的 .cube 3D LUT")
    parser.add_argument("--size", type=int, default=LUT_SIZE, help="LUT 边长，默认 33")
    parser.add_argument("--preview", type=Path, default=None, help="另外写一张色块对比图")
    args = parser.parse_args()
    if args.size < 2:
        raise SystemExit("--size 至少为 2")

    print("正在建立 sRGB 色域边界表…")
    srgb_max_chroma(0.5, 0.0)

    paths: list[Path] = []
    for mode in MODES:
        path = OUT_DIR / mode["filename"]
        print(f"写入 {path.name} …")
        write_cube(path, mode["title"], mode["comment"], args.size, mode["fn"])
        paths.append(path)

    print("检查格点与现象…")
    self_check(paths, args.size)
    diagnose()
    if args.preview is not None:
        write_preview(args.preview)
        print(f"预览图：{args.preview}")
    print("\n已写到：")
    for mode in MODES:
        print(f"  {mode['filename']}")
        print(f"    {mode['comment']}")


if __name__ == "__main__":
    main()
