import argparse
import json
import os
import random
from pathlib import Path


def _to_yolo_segmentation(segmentation, img_w, img_h):
    yolo_segments = []
    for poly in segmentation:
        if not isinstance(poly, list):
            continue
        if len(poly) < 6:
            continue
        coords = []
        for i in range(0, len(poly), 2):
            x = poly[i] / img_w
            y = poly[i + 1] / img_h
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        yolo_segments.append(" ".join(coords))
    return yolo_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--out-dir", required=True, help="Output dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    names = [c["name"] for c in categories]

    images = {img["id"]: img for img in coco["images"]}
    ann_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    out_dir = Path(args.out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    skipped_rle = 0

    for img_id, img in images.items():
        file_name = img["file_name"]
        img_w = img["width"]
        img_h = img["height"]
        img_path = Path(args.images_dir) / file_name
        image_paths.append(img_path)

        label_path = labels_dir / (Path(file_name).stem + ".txt")
        label_lines = []
        for ann in ann_by_image.get(img_id, []):
            segmentation = ann.get("segmentation", [])
            if not segmentation:
                continue
            if isinstance(segmentation, dict):
                skipped_rle += 1
                continue
            yolo_segments = _to_yolo_segmentation(segmentation, img_w, img_h)
            class_idx = id_to_idx.get(ann["category_id"], None)
            if class_idx is None:
                continue
            for seg in yolo_segments:
                label_lines.append(f"{class_idx} {seg}")

        if label_lines:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    random.seed(args.seed)
    random.shuffle(image_paths)

    val_count = max(1, int(len(image_paths) * args.val_ratio))
    val_set = set(image_paths[:val_count])

    train_list = out_dir / "train.txt"
    val_list = out_dir / "val.txt"

    with train_list.open("w", encoding="utf-8") as f_train, val_list.open("w", encoding="utf-8") as f_val:
        for p in image_paths:
            p_str = str(p.resolve())
            if p in val_set:
                f_val.write(p_str + "\n")
            else:
                f_train.write(p_str + "\n")

    data_yaml = out_dir / "data.yaml"
    yaml_lines = [
        f"path: {out_dir}",
        "train: train.txt",
        "val: val.txt",
        "task: segment",
        "names:",
    ]
    for i, name in enumerate(names):
        yaml_lines.append(f"  {i}: {name}")
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    if skipped_rle:
        print(f"Warning: skipped {skipped_rle} RLE segmentations.")
    print(f"Wrote labels to {labels_dir}")
    print(f"Wrote dataset config to {data_yaml}")


if __name__ == "__main__":
    main()
