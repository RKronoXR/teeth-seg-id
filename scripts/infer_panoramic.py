import argparse
import json
from pathlib import Path

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


def load_image(path, img_size):
    original = Image.open(path).convert("RGB")
    original_size = original.size

    if img_size > 0:
        image = original.resize((img_size, img_size), Image.BILINEAR)
    else:
        image = original

    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    return image_tensor, image_np, original_size, image.size


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


def make_overlay(image_np, prediction, threshold, show_scores, output_path):
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(image_np)
    ax.axis("off")
    ax.set_title("Tooth segmentation and FDI identification", fontsize=12, pad=18)

    masks = prediction["masks"].squeeze(1).detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    kept = 0

    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
        if score < threshold:
            continue

        binary_mask = mask >= 0.5
        if binary_mask.sum() == 0:
            continue

        overlay = np.zeros((*binary_mask.shape, 4), dtype=float)
        overlay[..., 0] = (i * 37 % 255) / 255
        overlay[..., 1] = (i * 67 % 255) / 255
        overlay[..., 2] = (i * 97 % 255) / 255
        overlay[..., 3] = binary_mask * 0.35
        ax.imshow(overlay)

        x1, y1, x2, y2 = box
        fdi = label_to_fdi(label)

        text = str(fdi)
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

        kept += 1

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    return kept


def prediction_to_json(image_path, prediction, threshold, original_size, model_input_size):
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    masks = prediction["masks"].squeeze(1).detach().cpu().numpy()

    items = []

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < threshold:
            continue

        binary_mask = mask >= 0.5
        area = int(binary_mask.sum())

        if area == 0:
            continue

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
        "threshold": threshold,
        "n_predictions": len(items),
        "predictions": items,
    }


def infer_one(model, image_path, output_dir, device, img_size, threshold, show_scores):
    image_tensor, image_np, original_size, model_input_size = load_image(image_path, img_size)

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    stem = Path(image_path).stem
    overlay_path = output_dir / f"{stem}_prediction.png"
    json_path = output_dir / f"{stem}_prediction.json"

    kept = make_overlay(
        image_np=image_np,
        prediction=prediction,
        threshold=threshold,
        show_scores=show_scores,
        output_path=overlay_path,
    )

    result = prediction_to_json(
        image_path=image_path,
        prediction=prediction,
        threshold=threshold,
        original_size=original_size,
        model_input_size=model_input_size,
    )

    json_path.write_text(json.dumps(result, indent=2))

    print(f"Image: {image_path}")
    print(f"Predictions kept: {kept}")
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
            img_size=args.img_size,
            threshold=args.threshold,
            show_scores=args.show_scores,
        )


if __name__ == "__main__":
    main()
