#!/usr/bin/env python3
"""Prepare Open Images (v6) subset and export to YOLO format.

This script uses FiftyOne to download a subset of Open Images and export
it as a YOLOv5/YOLOv8 compatible dataset (images + labels + data yaml).

Usage examples:
  python scripts/prepare_openimages_v6.py --classes person,dog,cat,car,bicycle --max-samples 20000 --output datasets/openimages_v6

If you prefer to only create the YAML and not download, pass `--dry-run`.
"""
import argparse
import os
import sys
import difflib


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Download Open Images subset and export to YOLO format")
    parser.add_argument("--classes", type=str, default="",
                        help="Comma-separated class list (use names as in Open Images). If empty, --num-classes will be used to auto-select classes.")
    parser.add_argument("--num-classes", type=int, default=10, help="If --classes is empty, auto-select this many classes from Open Images (max 100)")
    parser.add_argument("--max-samples", type=int, default=20000, help="Total number of images to sample")
    parser.add_argument("--output", type=str, default="datasets/openimages_v6", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't download; just create YAML and show commands")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    outdir = args.output
    ensure_dir(outdir)

    data_yaml = os.path.join(outdir, "openimages_v6.yaml")
    print(f"Preparing Open Images subset with classes: {classes}")
    print(f"Target sample size: {args.max_samples}")
    print(f"Output directory: {outdir}")

    # Create a small dataset YAML (YOLO format) that points to train/val folders
    yaml_content = {
        "path": os.path.abspath(outdir),
        "train": "train/images",
        "val": "val/images",
        "names": classes,
    }

    # Write YAML
    try:
        import yaml
        ensure_dir(os.path.dirname(data_yaml))
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote dataset YAML: {data_yaml}")
    except Exception as e:
        print("Failed to write YAML (yaml missing?):", e)

    if args.dry_run:
        print("Dry run - not downloading. To download, re-run without --dry-run.")
        print("Recommended command to download and export using FiftyOne:")
        print("python -c \"from fiftyone.zoo import load_zoo_dataset; ds=load_zoo_dataset('open-images-v6', split='train', classes=%r, max_samples=%d); ds.export(export_dir='%s', dataset_type=fiftyone.types.YOLOv5Dataset, label_field='ground_truth')\"" % (classes, args.max_samples, os.path.join(outdir, 'train')))
        return

    # Attempt to use FiftyOne to download and export
    try:
        import fiftyone as fo
        from fiftyone.zoo import load_zoo_dataset
        from fiftyone import types as fot
    except Exception as e:
        print("Missing FiftyOne or related libs. Install with: pip install fiftyone")
        sys.exit(1)

    # Get valid Open Images class names and map requested classes to valid names
    try:
        from fiftyone.utils import openimages as fo_oi
        valid_classes = fo_oi.get_classes()
    except Exception:
        try:
            import fiftyone.utils.openimages as fo_oi2
            valid_classes = fo_oi2.get_classes()
        except Exception:
            valid_classes = []

    if not valid_classes and not classes:
        print("Error: could not fetch Open Images class list and no classes were provided. Aborting.")
        sys.exit(1)

    def match_class(name, candidates):
        if not candidates:
            return name
        name_l = name.lower()
        for c in candidates:
            if c.lower() == name_l:
                return c
        for c in candidates:
            if name_l in c.lower():
                return c
        close = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if close:
            return close[0]
        return None

    # Auto-select classes if none provided
    if not classes:
        num = max(1, min(100, int(args.num_classes)))
        classes = valid_classes[:num]
        print(f"Auto-selected {len(classes)} classes from Open Images")
    else:
        mapped = []
        for name in classes:
            m = match_class(name, valid_classes)
            if m:
                mapped.append(m)
            else:
                print(f"Warning: could not match class '{name}' to Open Images; using literal name")
                mapped.append(name)
        # Fill up to num-classes if requested
        if args.num_classes and len(mapped) < args.num_classes and valid_classes:
            needed = args.num_classes - len(mapped)
            for c in valid_classes:
                if c not in mapped:
                    mapped.append(c)
                    needed -= 1
                    if needed <= 0:
                        break
        classes = mapped

    per_class = max(1, args.max_samples // max(1, len(classes)))
    print(f"Per-class target (approx): {per_class}")

    # Load samples from Open Images using FiftyOne's zoo loader
    print("Starting download from FiftyOne open-images-v6 zoo... this can take a while and requires internet and disk space.")
    ds = load_zoo_dataset(
        "open-images-v6",
        split="train",
        classes=classes,
        max_samples=args.max_samples,
        label_types=["detections"],
    )

    print(f"Downloaded dataset with {len(ds)} samples")

    # Determine label field name
    # Common label fields: "ground_truth" or "detections" depending on loader
    label_field = "ground_truth" if "ground_truth" in ds.get_field_schema() else "detections"
    print(f"Using label field: {label_field}")

    # Export to YOLO format (YOLOv5 format compatible with YOLOv8)
    export_dir = os.path.join(outdir)
    print(f"Exporting dataset to YOLO format at {export_dir} ...")
    try:
        ds.export(export_dir=export_dir, dataset_type=fot.YOLOv5Dataset, label_field=label_field, classes=classes)
    except Exception as e:
        print("Export failed:", e)
        print("As a fallback, try exporting using FiftyOne GUI or check label_field value.")
        sys.exit(1)

    print("Export complete.")
    print("Suggested next steps:")
    print(f"- Train with ultralytics: python -m ultralytics.train data={data_yaml} model=yolov8n.pt imgsz=640 epochs=50 batch=16")


if __name__ == "__main__":
    main()
