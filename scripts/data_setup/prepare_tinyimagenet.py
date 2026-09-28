"""
Restructure TinyImageNet-200 into ImageFolder format.

Source layout (after unzipping tiny-imagenet-200.zip):
    tiny-imagenet-200/
        train/<wnid>/images/*.JPEG          # nested under images/
        train/<wnid>/<wnid>_boxes.txt
        val/images/*.JPEG                    # flat; labels in val_annotations.txt
        val/val_annotations.txt              # filename<TAB>wnid<TAB>...
        test/images/*.JPEG                   # no labels (unused)
        wnids.txt
        words.txt

Target layout (canonical for our pipeline):
    data/tinyimagenet/
        train/<wnid>/*.JPEG
        val/<wnid>/*.JPEG

Idempotent: safe to re-run if interrupted.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def flatten_train(src_train: Path) -> int:
    """Move train/<wnid>/images/*.JPEG up to train/<wnid>/*.JPEG, drop boxes.txt."""
    moved = 0
    for wnid_dir in sorted(src_train.iterdir()):
        if not wnid_dir.is_dir():
            continue
        images_dir = wnid_dir / "images"
        if not images_dir.exists():
            continue
        for jpeg in images_dir.glob("*.JPEG"):
            target = wnid_dir / jpeg.name
            if not target.exists():
                shutil.move(str(jpeg), str(target))
                moved += 1
        if not any(images_dir.iterdir()):
            images_dir.rmdir()
        boxes_file = wnid_dir / f"{wnid_dir.name}_boxes.txt"
        if boxes_file.exists():
            boxes_file.unlink()
    return moved


def restructure_val(src_val: Path) -> int:
    """Move val/images/*.JPEG into val/<wnid>/<img>.JPEG using val_annotations.txt."""
    annotations = src_val / "val_annotations.txt"
    flat_images = src_val / "images"
    if not annotations.exists():
        return 0  # already restructured
    if not flat_images.exists():
        return 0
    moved = 0
    with open(annotations, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            class_dir = src_val / wnid
            class_dir.mkdir(exist_ok=True)
            src_img = flat_images / img_name
            dst_img = class_dir / img_name
            if src_img.exists() and not dst_img.exists():
                shutil.move(str(src_img), str(dst_img))
                moved += 1
    if flat_images.exists() and not any(flat_images.iterdir()):
        flat_images.rmdir()
    if annotations.exists():
        annotations.unlink()
    return moved


def materialize(src_root: Path, dst_root: Path) -> None:
    """Move train/ and val/ from src_root into dst_root (or symlink if same disk)."""
    for split in ("train", "val"):
        src = src_root / split
        dst = dst_root / split
        if dst.exists():
            print(f"  {dst} already exists, skipping move")
            continue
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  moved {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "tinyimagenet"),
        help="Target dataset directory (will contain train/ and val/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    src_root = data_dir / "tiny-imagenet-200"

    if not src_root.exists() and (data_dir / "train").exists():
        print(f"[info] Already restructured at {data_dir}/train and {data_dir}/val")
        sys.exit(0)

    if not src_root.exists():
        print(f"[error] Source not found: {src_root}", file=sys.stderr)
        print("Did you run: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip ?", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] Flattening train/<wnid>/images -> train/<wnid>/")
    moved_train = flatten_train(src_root / "train")
    print(f"      moved {moved_train} train images")

    print(f"[2/3] Restructuring val/images -> val/<wnid>/")
    moved_val = restructure_val(src_root / "val")
    print(f"      moved {moved_val} val images")

    print(f"[3/3] Promoting train/ and val/ to {data_dir}/")
    materialize(src_root, data_dir)

    test_dir = src_root / "test"
    if test_dir.exists():
        shutil.rmtree(str(test_dir))
        print(f"      removed unused test/ (no labels)")

    if src_root.exists() and not any((src_root / sub).exists() for sub in ("train", "val", "test")):
        for leftover in ("wnids.txt", "words.txt"):
            f = src_root / leftover
            if f.exists():
                shutil.move(str(f), str(data_dir / leftover))
        try:
            src_root.rmdir()
            print(f"      cleaned up {src_root}")
        except OSError:
            pass

    print()
    print("Done. Final structure:")
    for split in ("train", "val"):
        split_dir = data_dir / split
        if split_dir.exists():
            n_classes = sum(1 for p in split_dir.iterdir() if p.is_dir())
            print(f"  {split_dir}: {n_classes} classes")


if __name__ == "__main__":
    main()
