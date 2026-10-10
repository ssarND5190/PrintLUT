# -*- coding: utf-8 -*-
"""用 FOGRA39（ISO Coated v2）生成屏幕软打样 3D LUT。

特性文件是 ECI 的 ISO Coated v2（ISOcoated_v2_eci.icc），
测量数据为 FOGRA39L，对应 ISO 12647-2:2004 涂布纸胶印，总墨量 330%。
转换使用 LittleCMS 的软打样：sRGB → 印刷特性文件 → sRGB。
显示器按 sRGB 显示这些数字时，看到的就是这份印刷条件的软打样。

  FOGRA39-相对比色.cube   纸白对齐到屏幕白。色域内的颜色保持比色一致，
                          色域外被裁进涂布纸色域。印刷黑大约停在 L* 11，不会黑到屏幕黑。
  FOGRA39-感知意图.cube   用特性文件里的感知意图把整个画面压进色域，适合照片。
  FOGRA39-模拟纸白.cube   绝对比色。纸白呈现为略冷的浅灰，黑是油墨黑。

特性文件放在 profiles/ISOcoated_v2_eci.icc，来自 ECI Offset 2009：
https://www.eci.org/doku.php?id=en:downloads

用法：
  python generate_fogra39_softproof.py
  python generate_fogra39_softproof.py --preview preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw
from PIL.ImageCms import Flags, Intent

LUT_SIZE = 33
ROOT = Path(__file__).resolve().parent
PROFILE_PATH = ROOT / "profiles" / "ISOcoated_v2_eci.icc"

# (文件名, 标题, 说明, 输入→印刷的意图, 印刷→屏幕的意图)
MODES = (
    {
        "filename": "FOGRA39-相对比色.cube",
        "title": "FOGRA39 relative colorimetric",
        "comment": "ISO Coated v2 / FOGRA39L soft proof, relative colorimetric. Paper white maps to display white.",
        "intent": Intent.RELATIVE_COLORIMETRIC,
        "proof_intent": Intent.RELATIVE_COLORIMETRIC,
    },
    {
        "filename": "FOGRA39-感知意图.cube",
        "title": "FOGRA39 perceptual",
        "comment": "ISO Coated v2 / FOGRA39L soft proof, perceptual intent into the coated gamut.",
        "intent": Intent.PERCEPTUAL,
        "proof_intent": Intent.RELATIVE_COLORIMETRIC,
    },
    {
        "filename": "FOGRA39-模拟纸白.cube",
        "title": "FOGRA39 simulate paper",
        "comment": "ISO Coated v2 / FOGRA39L soft proof, absolute colorimetric, paper white and ink black simulated.",
        "intent": Intent.RELATIVE_COLORIMETRIC,
        "proof_intent": Intent.ABSOLUTE_COLORIMETRIC,
    },
)


def load_profiles():
    if not PROFILE_PATH.is_file():
        raise SystemExit(
            f"找不到 {PROFILE_PATH}\n"
            "请从 https://www.eci.org/doku.php?id=en:downloads 下载 eci_offset_2009.zip，"
            "并把其中的 ISOcoated_v2_eci.icc 放到 profiles 目录。"
        )
    proof = ImageCms.getOpenProfile(str(PROFILE_PATH))
    description = proof.profile.profile_description
    if "Coated v2" not in description:
        raise SystemExit(f"特性文件不是 ISO Coated v2：{description}")
    srgb = ImageCms.createProfile("sRGB")
    return srgb, proof, description


def build_transform(srgb, proof, mode):
    return ImageCms.buildProofTransform(
        srgb,
        srgb,
        proof,
        "RGB",
        "RGB",
        mode["intent"],
        mode["proof_intent"],
        Flags.SOFTPROOFING,
    )


def lattice_image(size: int) -> Image.Image:
    """红通道变化最快，与 Adobe .cube / dwm_lut 的格点顺序一致。"""
    denom = size - 1
    image = Image.new("RGB", (size * size, size))
    pixels = image.load()
    for b in range(size):
        bb = int(round(b * 255 / denom))
        for g in range(size):
            gg = int(round(g * 255 / denom))
            x0 = g * size
            for r in range(size):
                pixels[x0 + r, b] = (int(round(r * 255 / denom)), gg, bb)
    return image


def transform_rgb(transform, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    image = Image.new("RGB", (1, 1), rgb)
    return ImageCms.applyTransform(image, transform).getpixel((0, 0))


def sample_lut(transform, size: int) -> list[tuple[int, int, int]]:
    output = ImageCms.applyTransform(lattice_image(size), transform)
    pixels = output.load()
    values: list[tuple[int, int, int]] = []
    width = size * size
    for b in range(size):
        for g in range(size):
            x0 = g * size
            for r in range(size):
                values.append(pixels[x0 + r, b])
    return values


def write_cube(path: Path, title: str, comment: str, size: int, values: list[tuple[int, int, int]]) -> None:
    # 中文说明放在 LUT_3D_SIZE 之前。dwm_lut 读到尺寸之后只接受以数字开头的行。
    header = [
        f'TITLE "{title}"',
        f"# {comment}",
        "# Soft proof sampled with LittleCMS from ISOcoated_v2_eci.icc (FOGRA39L).",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        f"LUT_3D_SIZE {size}",
    ]
    body = [f"{r / 255:.6f} {g / 255:.6f} {b / 255:.6f}" for r, g, b in values]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(header))
        handle.write("\n")
        handle.write("\n".join(body))
        handle.write("\n")


def node_index(size: int, r: int, g: int, b: int) -> int:
    return (b * size + g) * size + r


def read_nodes(path: Path) -> tuple[int, list[tuple[float, float, float]]]:
    size = None
    nodes: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("LUT_3D_SIZE"):
            size = int(text.split()[-1])
            continue
        if not text or text.startswith("#") or text.startswith("TITLE") or text.startswith("DOMAIN"):
            continue
        parts = text.split()
        if len(parts) == 3:
            nodes.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if size is None:
        raise RuntimeError(f"{path.name} 缺少 LUT_3D_SIZE")
    return size, nodes


def self_check(srgb, proof, paths: list[Path], size: int) -> None:
    transforms = [build_transform(srgb, proof, mode) for mode in MODES]
    colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "skin": (232, 191, 166),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
    }
    got = {
        name: [transform_rgb(transform, colors[key]) for key in colors]
        for name, transform in zip(("rel", "per", "paper"), transforms)
    }
    rel = dict(zip(colors, got["rel"]))
    per = dict(zip(colors, got["per"]))
    pap = dict(zip(colors, got["paper"]))

    assert rel["white"] == (255, 255, 255), rel["white"]
    assert rel["gray"] == (128, 128, 128), rel["gray"]
    assert rel["skin"] == (232, 191, 166), rel["skin"]
    assert rel["black"][0] > 20 and max(rel["black"]) < 50, rel["black"]
    assert rel["blue"][2] < 180 and rel["blue"][0] > 40, rel["blue"]
    assert rel["green"][1] < 210 and rel["green"][2] < 90, rel["green"]
    assert rel["yellow"][0] > 240 and rel["yellow"][2] < 80, rel["yellow"]
    assert per["gray"] != colors["gray"]
    assert per["white"] == (255, 255, 255)
    assert pap["white"] != (255, 255, 255)
    assert pap["white"][2] > pap["white"][0] > 220, pap["white"]
    assert min(pap["black"]) > 15, pap["black"]

    # 0.5 落在 33 级格子的正中间，对应 8-bit 128。
    mid = size // 2
    expect = {
        "FOGRA39-相对比色.cube": {
            (0, 0, 0): rel["black"],
            (size - 1, size - 1, size - 1): rel["white"],
            (mid, mid, mid): rel["gray"],
            (0, 0, size - 1): rel["blue"],
        },
        "FOGRA39-感知意图.cube": {
            (size - 1, size - 1, size - 1): per["white"],
            (0, 0, size - 1): per["blue"],
        },
        "FOGRA39-模拟纸白.cube": {
            (size - 1, size - 1, size - 1): pap["white"],
            (0, 0, 0): pap["black"],
        },
    }
    for path in paths:
        got_size, nodes = read_nodes(path)
        assert got_size == size
        assert len(nodes) == size ** 3, (path.name, len(nodes))
        for (r, g, b), rgb in expect[path.name].items():
            node = nodes[node_index(size, r, g, b)]
            target = tuple(channel / 255 for channel in rgb)
            assert all(abs(a - b) < 1e-5 for a, b in zip(node, target)), (path.name, rgb, node)


def diagnose(transforms) -> None:
    swatches = (
        ("红", (255, 0, 0)),
        ("绿", (0, 255, 0)),
        ("蓝", (0, 0, 255)),
        ("青", (0, 255, 255)),
        ("品红", (255, 0, 255)),
        ("黄", (255, 255, 0)),
        ("肤色", (232, 191, 166)),
        ("青绿", (0, 128, 128)),
        ("灰", (128, 128, 128)),
        ("白", (255, 255, 255)),
        ("黑", (0, 0, 0)),
    )
    names = ("相对", "感知", "纸白")
    print("\n输入 RGB255  →  FOGRA39 软打样")
    for name, rgb in swatches:
        cells = [str(transform_rgb(transform, rgb)) for transform in transforms]
        print(f"{name:<4} {rgb!s:<18} " + "  ".join(f"{label}:{cell}" for label, cell in zip(names, cells)))


def write_preview(path: Path, transforms) -> None:
    swatches = (
        ("R", (255, 0, 0)),
        ("G", (0, 255, 0)),
        ("B", (0, 0, 255)),
        ("C", (0, 255, 255)),
        ("M", (255, 0, 255)),
        ("Y", (255, 255, 0)),
        ("skin", (232, 191, 166)),
        ("teal", (0, 128, 128)),
        ("gray", (128, 128, 128)),
        ("white", (255, 255, 255)),
        ("black", (0, 0, 0)),
    )
    columns = ("source", "relative", "perceptual", "paper")
    pad, label_w, cell, head = 8, 52, 78, 22
    image = Image.new("RGB", (pad + label_w + cell * len(columns) + pad, pad + head + cell * len(swatches) + pad), (32, 32, 32))
    draw = ImageDraw.Draw(image)
    for index, title in enumerate(columns):
        draw.text((pad + label_w + index * cell + 4, pad), title, fill=(230, 230, 230))
    for row, (name, rgb) in enumerate(swatches):
        y = pad + head + row * cell
        draw.text((pad, y + cell // 2 - 6), name, fill=(220, 220, 220))
        colors = [rgb, *(transform_rgb(transform, rgb) for transform in transforms)]
        for index, color in enumerate(colors):
            x = pad + label_w + index * cell
            draw.rectangle((x + 4, y + 4, x + cell - 4, y + cell - 4), fill=color)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 FOGRA39 / ISO Coated v2 生成软打样 .cube")
    parser.add_argument("--size", type=int, default=LUT_SIZE)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    if args.size < 2:
        raise SystemExit("--size 至少为 2")

    srgb, proof, description = load_profiles()
    print(f"特性文件：{description}  ({PROFILE_PATH.name})")

    transforms = []
    paths: list[Path] = []
    for mode in MODES:
        transform = build_transform(srgb, proof, mode)
        transforms.append(transform)
        print(f"采样 {mode['filename']} …")
        values = sample_lut(transform, args.size)
        path = ROOT / mode["filename"]
        write_cube(path, mode["title"], mode["comment"], args.size, values)
        paths.append(path)

    print("检查格点…")
    self_check(srgb, proof, paths, args.size)
    diagnose(transforms)
    if args.preview is not None:
        write_preview(args.preview, transforms)
        print(f"预览图：{args.preview}")
    print("\n已写到：")
    for mode in MODES:
        print(f"  {mode['filename']}")
        print(f"    {mode['comment']}")


if __name__ == "__main__":
    main()
