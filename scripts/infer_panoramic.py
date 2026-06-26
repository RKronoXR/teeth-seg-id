import argparse
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


def image_to_tensor_and_np(image):
    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return image_tensor, image_np


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

    if img_size > 0:
        inference_image = inference_image.resize((img_size, img_size), Image.BILINEAR)
        display_image = display_image.resize((img_size, img_size), Image.BILINEAR)

    image_tensor, inference_np = image_to_tensor_and_np(inference_image)
    display_np = np.asarray(display_image).astype(np.float32) / 255.0

    return image_tensor, inference_np, display_np, original_size, inference_image.size


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


def prediction_to_json(image_path, prediction, kept_indices, args, original_size, model_input_size):
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
        })

    return {
        "image": str(image_path),
        "original_size_xy": list(original_size),
        "model_input_size_xy": list(model_input_size),
        "preprocess": args.preprocess,
        "display": "preprocessed" if args.display_preprocessed else "original",
        "threshold": args.threshold,
        "min_mask_area": args.min_mask_area,
        "keep_best_per_fdi": args.keep_best_per_fdi,
        "n_predictions": len(items),
        "predictions": items,
    }


def infer_one(model, image_path, output_dir, device, args):
    image_tensor, inference_np, display_np, original_size, model_input_size = load_image(
        path=image_path,
        img_size=args.img_size,
        preprocess=args.preprocess,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        display_preprocessed=args.display_preprocessed,
    )

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    kept_indices = get_kept_indices(
        prediction=prediction,
        threshold=args.threshold,
        min_mask_area=args.min_mask_area,
        keep_best_per_fdi=args.keep_best_per_fdi,
    )

    stem = Path(image_path).stem
    display_name = "display_preprocessed" if args.display_preprocessed else "display_original"
    suffix = f"{args.preprocess}_thr{args.threshold}_{display_name}"
    overlay_path = output_dir / f"{stem}_{suffix}_prediction.png"
    json_path = output_dir / f"{stem}_{suffix}_prediction.json"

    make_overlay(
        display_np=display_np,
        prediction=prediction,
        kept_indices=kept_indices,
        show_scores=args.show_scores,
        output_path=overlay_path,
    )

    result = prediction_to_json(
        image_path=image_path,
        prediction=prediction,
        kept_indices=kept_indices,
        args=args,
        original_size=original_size,
        model_input_size=model_input_size,
    )

    json_path.write_text(json.dumps(result, indent=2))

    print(f"Image: {image_path}")
    print(f"Preprocess for inference: {args.preprocess}")
    print(f"Display image: {'preprocessed' if args.display_preprocessed else 'original'}")
    print(f"Predictions kept: {len(kept_indices)}")
    print(f"Saved image: {overlay_path}")
    print(f"Saved JSON: {json_path}")


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
        description="Run tooth segmentation and FDI identification on panoramic radiographs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", default=None, help="Path to one panoramic image.")
    source.add_argument("--image-dir", default=None, help="Path to a folder with panoramic images.")

    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", default="outputs/predictions/external", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction score threshold.")
    parser.add_argument("--img-size", type=int, default=640, help="Resize image to this square size. Use 0 to keep original size.")
    parser.add_argument("--show-scores", action="store_true", help="Show confidence score below FDI label.")
    parser.add_argument("--min-mask-area", type=int, default=100, help="Discard predictions with mask area below this value.")
    parser.add_argument("--keep-best-per-fdi", action="store_true", help="Keep only the highest-scoring prediction for each FDI tooth number.")
    parser.add_argument("--preprocess", choices=["none", "clahe", "equalize"], default="none", help="Optional preprocessing for model inference.")
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
