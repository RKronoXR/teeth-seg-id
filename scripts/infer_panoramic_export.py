import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn


FDI_CLASSES = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)


def label_to_fdi(label):
    return FDI_CLASSES[int(label) - 1]


def preprocess_image(original, method, clahe_clip, clahe_grid):
    gray = np.asarray(original.convert("L"))

    if method == "none":
        out = gray
    elif method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        out = clahe.apply(gray)
    elif method == "equalize":
        out = cv2.equalizeHist(gray)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    return Image.fromarray(out).convert("RGB")


def letterbox_image(image, img_size):
    original_w, original_h = image.size

    if img_size <= 0:
        return image, {
            "resize_mode": "original",
            "scale": 1.0,
            "pad_x": 0,
            "pad_y": 0,
            "resized_size_xy": [original_w, original_h],
            "model_input_size_xy": [original_w, original_h],
        }

    scale = min(img_size / original_w, img_size / original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), color=(0, 0, 0))

    pad_x = (img_size - resized_w) // 2
    pad_y = (img_size - resized_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, {
        "resize_mode": "letterbox",
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "resized_size_xy": [resized_w, resized_h],
        "model_input_size_xy": [img_size, img_size],
    }


def load_image(path, img_size, preprocess, clahe_clip, clahe_grid, display_preprocessed):
    original = Image.open(path).convert("RGB")
    original_size = original.size

    inference_image = preprocess_image(
        original=original,
        method=preprocess,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
    )

    display_image = inference_image if display_preprocessed else original
    inference_image, resize_metadata = letterbox_image(inference_image, img_size)

    inference_np = np.asarray(inference_image).astype(np.float32) / 255.0
    display_np = np.asarray(display_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(inference_np).permute(2, 0, 1)

    resize_metadata["original_size_xy"] = list(original_size)

    return image_tensor, display_np, original_size, inference_image.size, resize_metadata


def restore_prediction_to_original(prediction, original_size, resize_metadata):
    if resize_metadata["resize_mode"] == "original":
        return prediction

    original_w, original_h = original_size
    scale = resize_metadata["scale"]
    pad_x = resize_metadata["pad_x"]
    pad_y = resize_metadata["pad_y"]
    resized_w, resized_h = resize_metadata["resized_size_xy"]

    restored = dict(prediction)

    boxes = prediction["boxes"].detach().cpu().clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_h)
    restored["boxes"] = boxes

    masks = prediction["masks"].detach().cpu().numpy()
    restored_masks = []

    for mask in masks:
        mask_2d = mask[0]
        cropped = mask_2d[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
        resized = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        restored_masks.append(resized[None, :, :])

    if restored_masks:
        restored["masks"] = torch.from_numpy(np.stack(restored_masks)).float()
    else:
        restored["masks"] = torch.empty((0, 1, original_h, original_w), dtype=torch.float32)

    restored["labels"] = prediction["labels"].detach().cpu()
    restored["scores"] = prediction["scores"].detach().cpu()

    return restored


def load_model(checkpoint_path, device):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=33,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def get_kept_indices(prediction, threshold, min_mask_area, keep_best_per_fdi):
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    masks = prediction["masks"].squeeze(1).detach().cpu().numpy()

    candidates = []

    for i, (label, score, mask) in enumerate(zip(labels, scores, masks)):
        binary_mask = mask >= 0.5
        area = int(binary_mask.sum())

        if score < threshold:
            continue

        if area < min_mask_area:
            continue

        candidates.append({
            "index": i,
            "fdi": label_to_fdi(label),
            "score": float(score),
            "area": area,
        })

    if not keep_best_per_fdi:
        return [c["index"] for c in candidates]

    best_by_fdi = {}
    for c in candidates:
        fdi = c["fdi"]
        if fdi not in best_by_fdi or c["score"] > best_by_fdi[fdi]["score"]:
            best_by_fdi[fdi] = c

    kept = sorted(best_by_fdi.values(), key=lambda x: x["fdi"])
    return [c["index"] for c in kept]


def save_masks(prediction, kept_indices, masks_dir, stem):
    masks = prediction["masks"].squeeze(1).detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    mask_paths = {}

    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in kept_indices:
        fdi = label_to_fdi(labels[i])
        score = float(scores[i])
        binary_mask = (masks[i] >= 0.5).astype(np.uint8) * 255

        score_text = f"{score:.3f}".replace(".", "p")
        mask_path = masks_dir / f"{stem}_FDI_{fdi}_score_{score_text}_mask.png"

        Image.fromarray(binary_mask).save(mask_path)
        mask_paths[i] = str(mask_path)

    return mask_paths


def make_overlay(display_np, prediction, kept_indices, show_scores, output_path):
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(display_np)
    ax.axis("off")
    ax.set_title("Tooth segmentation and FDI identification", fontsize=12, pad=18)

    masks = prediction["masks"].squeeze(1).detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    for rank, i in enumerate(kept_indices):
        mask = masks[i]
        box = boxes[i]
        label = labels[i]
        score = scores[i]

        binary_mask = mask >= 0.5
        if binary_mask.sum() == 0:
            continue

        overlay = np.zeros((*binary_mask.shape, 4), dtype=float)
        overlay[..., 0] = (rank * 37 % 255) / 255
        overlay[..., 1] = (rank * 67 % 255) / 255
        overlay[..., 2] = (rank * 97 % 255) / 255
        overlay[..., 3] = binary_mask * 0.35
        ax.imshow(overlay)

        x1, y1, x2, y2 = box
        text = str(label_to_fdi(label))

        if show_scores:
            text += f"\n{score:.2f}"

        ax.text(
            x1,
            y1,
            text,
            fontsize=8,
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.65, linewidth=0),
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def build_results(image_path, prediction, kept_indices, args, original_size, model_input_size, mask_paths, resize_metadata):
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    masks = prediction["masks"].squeeze(1).detach().cpu().numpy()

    items = []

    for i in kept_indices:
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        mask = masks[i]

        binary_mask = mask >= 0.5
        area = int(binary_mask.sum())

        ys, xs = np.where(binary_mask)
        centroid_x = float(xs.mean()) if len(xs) else None
        centroid_y = float(ys.mean()) if len(ys) else None

        x1, y1, x2, y2 = [float(v) for v in box]

        items.append({
            "fdi": int(label_to_fdi(label)),
            "class_id": int(label),
            "score": float(score),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            "mask_area_pixels": area,
            "centroid_xy": [centroid_x, centroid_y],
            "mask_path": mask_paths.get(i),
        })

    return {
        "image": str(image_path),
        "original_size_xy": list(original_size),
        "model_input_size_xy": list(model_input_size),
        "resize_metadata": resize_metadata,
        "preprocess": args.preprocess,
        "display": "preprocessed" if args.display_preprocessed else "original",
        "threshold": args.threshold,
        "min_mask_area": args.min_mask_area,
        "keep_best_per_fdi": args.keep_best_per_fdi,
        "n_predictions": len(items),
        "predictions": items,
    }


def save_csv(results, csv_path):
    rows = results["predictions"]

    fieldnames = [
        "fdi",
        "class_id",
        "score",
        "mask_area_pixels",
        "centroid_x",
        "centroid_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mask_path",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            cx, cy = row["centroid_xy"]

            writer.writerow({
                "fdi": row["fdi"],
                "class_id": row["class_id"],
                "score": row["score"],
                "mask_area_pixels": row["mask_area_pixels"],
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "mask_path": row["mask_path"],
            })


def save_report(results, overlay_path, json_path, csv_path, report_path):
    lines = []

    lines.append(f"# Panoramic inference report\n")
    lines.append(f"## Image\n")
    lines.append(f"- Source image: `{results['image']}`")
    lines.append(f"- Original size: `{results['original_size_xy']}`")
    lines.append(f"- Model input size: `{results['model_input_size_xy']}`")
    lines.append(f"- Preprocess: `{results['preprocess']}`")
    lines.append(f"- Display: `{results['display']}`")
    lines.append(f"- Threshold: `{results['threshold']}`")
    lines.append(f"- Minimum mask area: `{results['min_mask_area']}`")
    lines.append(f"- Keep best per FDI: `{results['keep_best_per_fdi']}`")
    lines.append(f"- Number of predictions: **{results['n_predictions']}**\n")

    lines.append("## Output files\n")
    lines.append(f"- Overlay image: `{overlay_path}`")
    lines.append(f"- JSON: `{json_path}`")
    lines.append(f"- CSV: `{csv_path}`\n")

    lines.append("## Predicted teeth\n")
    lines.append("| FDI | Score | Mask area | Centroid x | Centroid y | Mask file |")
    lines.append("|---:|---:|---:|---:|---:|---|")

    for item in results["predictions"]:
        cx, cy = item["centroid_xy"]
        lines.append(
            f"| {item['fdi']} | {item['score']:.3f} | "
            f"{item['mask_area_pixels']} | "
            f"{cx:.1f} | {cy:.1f} | `{item['mask_path']}` |"
        )

    report_path.write_text("\n".join(lines))


def infer_one(model, image_path, output_dir, device, args):
    image_tensor, display_np, original_size, model_input_size, resize_metadata = load_image(
        path=image_path,
        img_size=args.img_size,
        preprocess=args.preprocess,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        display_preprocessed=args.display_preprocessed,
    )

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    prediction = restore_prediction_to_original(
        prediction=prediction,
        original_size=original_size,
        resize_metadata=resize_metadata,
    )

    kept_indices = get_kept_indices(
        prediction=prediction,
        threshold=args.threshold,
        min_mask_area=args.min_mask_area,
        keep_best_per_fdi=args.keep_best_per_fdi,
    )

    stem = Path(image_path).stem
    display_name = "display_preprocessed" if args.display_preprocessed else "display_original"
    suffix = f"{args.preprocess}_thr{args.threshold}_{display_name}"

    case_dir = output_dir / stem
    masks_dir = case_dir / "masks"
    case_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = case_dir / f"{stem}_{suffix}_prediction.png"
    json_path = case_dir / f"{stem}_{suffix}_prediction.json"
    csv_path = case_dir / f"{stem}_{suffix}_prediction.csv"
    report_path = case_dir / f"{stem}_{suffix}_report.md"

    mask_paths = save_masks(
        prediction=prediction,
        kept_indices=kept_indices,
        masks_dir=masks_dir,
        stem=stem,
    )

    make_overlay(
        display_np=display_np,
        prediction=prediction,
        kept_indices=kept_indices,
        show_scores=args.show_scores,
        output_path=overlay_path,
    )

    results = build_results(
        image_path=image_path,
        prediction=prediction,
        kept_indices=kept_indices,
        args=args,
        original_size=original_size,
        model_input_size=model_input_size,
        mask_paths=mask_paths,
        resize_metadata=resize_metadata,
    )

    json_path.write_text(json.dumps(results, indent=2))
    save_csv(results, csv_path)
    save_report(results, overlay_path, json_path, csv_path, report_path)

    print(f"Image: {image_path}")
    print(f"Predictions kept: {len(kept_indices)}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved masks: {masks_dir}")


def collect_images(image, image_dir):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]

    if image:
        return [Path(image)]

    paths = []
    for ext in extensions:
        paths.extend(Path(image_dir).glob(ext))
        paths.extend(Path(image_dir).glob(ext.upper()))

    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Run panoramic tooth segmentation and export overlay, JSON, CSV, masks, and Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default=None, help="Path to one panoramic image.")
    source.add_argument("--image-dir", default=None, help="Path to a folder with panoramic images.")

    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", default="outputs/predictions/export", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.65, help="Prediction score threshold.")
    parser.add_argument("--img-size", type=int, default=640, help="Letterbox image to this model input size. Use 0 to keep original size.")
    parser.add_argument("--show-scores", action="store_true", help="Show confidence score below FDI label.")
    parser.add_argument("--min-mask-area", type=int, default=100, help="Discard predictions with mask area below this value.")
    parser.add_argument("--keep-best-per-fdi", action="store_true", help="Keep only the highest-scoring prediction for each FDI tooth number.")
    parser.add_argument("--preprocess", choices=["none", "clahe", "equalize"], default="clahe", help="Optional preprocessing for model inference.")
    parser.add_argument("--display-preprocessed", action="store_true", help="Draw results over the preprocessed image instead of the original image.")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    images = collect_images(args.image, args.image_dir)

    if not images:
        raise FileNotFoundError("No images found.")

    for image_path in images:
        infer_one(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
