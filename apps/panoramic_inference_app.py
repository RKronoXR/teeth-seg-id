import base64
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "app_uploads"
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "predictions"


def list_checkpoints():
    checkpoints = sorted(
        CHECKPOINT_ROOT.glob("*/best_model.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints


def save_uploaded_file(uploaded_file):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    path = UPLOAD_ROOT / f"{timestamp}_{safe_name}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def run_inference(
    image_path,
    checkpoint_path,
    threshold,
    min_mask_area,
    preprocess,
    keep_best_per_fdi,
    show_scores,
    display_preprocessed,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PREDICTION_ROOT / f"app_{Path(image_path).stem}_{timestamp}"

    cmd = [
        sys.executable,
        "scripts/infer_panoramic_export.py",
        "--image",
        str(image_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--threshold",
        str(threshold),
        "--min-mask-area",
        str(min_mask_area),
        "--preprocess",
        preprocess,
    ]

    if keep_best_per_fdi:
        cmd.append("--keep-best-per-fdi")

    if show_scores:
        cmd.append("--show-scores")

    if display_preprocessed:
        cmd.append("--display-preprocessed")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    return result, output_dir


def find_outputs(output_dir):
    pngs = sorted(output_dir.rglob("*_prediction.png"))
    jsons = sorted(output_dir.rglob("*_prediction.json"))
    csvs = sorted(output_dir.rglob("*_prediction.csv"))
    reports = sorted(output_dir.rglob("*_report.md"))

    return {
        "png": pngs[0] if pngs else None,
        "json": jsons[0] if jsons else None,
        "csv": csvs[0] if csvs else None,
        "report": reports[0] if reports else None,
    }


def create_output_zip(output_dir):
    zip_path = output_dir / f"{output_dir.name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                z.write(path, path.relative_to(output_dir))

    return zip_path


def make_preview_image(image_path, max_width):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if width <= max_width:
        return image

    new_height = int(height * (max_width / width))
    return image.resize((max_width, new_height), Image.LANCZOS)


def image_to_base64(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def render_prediction_image(image_path, display_mode, preview_width):
    encoded = image_to_base64(image_path)

    if display_mode == "Fixed width":
        box_style = f"width: {preview_width}px;"
        img_style = "width: 100%; max-width: 100%;"
    elif display_mode == "Fit page width":
        box_style = "width: 100%;"
        img_style = "width: 100%; max-width: 100%;"
    else:
        box_style = "width: 100%;"
        img_style = "width: auto; max-width: none;"

    html = f"""
    <div class="prediction-wrapper">
        <div class="prediction-scroll-box" style="{box_style}">
            <img src="data:image/png;base64,{encoded}" style="{img_style}">
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(
    page_title="Panoramic Tooth Segmentation",
    layout="wide",
)

st.markdown(
    """
    <style>
    .prediction-wrapper {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: flex-start;
    }
    .prediction-scroll-box {
        max-width: 100%;
        max-height: 78vh;
        overflow: auto;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 8px;
        padding: 8px;
        background: transparent;
    }
    .prediction-scroll-box img {
        display: block;
        height: auto;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Panoramic Tooth Segmentation and FDI Identification")

st.write(
    "Upload a panoramic radiograph, choose a model checkpoint, "
    "and run tooth segmentation with FDI numbering."
)

checkpoints = list_checkpoints()

if not checkpoints:
    st.error("No best_model.pth checkpoints found in outputs/checkpoints/.")
    st.stop()

checkpoint_labels = [str(p.relative_to(PROJECT_ROOT)) for p in checkpoints]

selected_checkpoint_label = st.selectbox(
    "Model checkpoint",
    checkpoint_labels,
    index=0,
)

checkpoint_path = PROJECT_ROOT / selected_checkpoint_label

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload panoramic image",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    )

with col2:
    manual_path = st.text_input(
        "Or use image path on Spark",
        value="",
        placeholder="/home/rkronoxr/Downloads/mi_panoramica.jpg",
    )

st.subheader("Inference options")

col1, col2, col3 = st.columns(3)

with col1:
    threshold = st.slider("Threshold", 0.05, 0.95, 0.65, 0.05)
    min_mask_area = st.number_input("Minimum mask area", min_value=0, value=100, step=50)

with col2:
    preprocess = st.selectbox("Preprocess for inference", ["none", "clahe", "equalize"], index=1)
    keep_best_per_fdi = st.checkbox("Keep best per FDI", value=True)

with col3:
    show_scores = st.checkbox("Show scores", value=True)
    display_preprocessed = st.checkbox("Display preprocessed image", value=False)

st.subheader("Display options")

display_col1, display_col2 = st.columns(2)

with display_col1:
    display_mode = st.selectbox(
        "Prediction display mode",
        ["Fixed width", "Fit page width", "Scrollable original"],
        index=0,
    )

with display_col2:
    preview_width = st.slider("Preview width (px)", 400, 1600, 850, 50)

run_button = st.button("Run inference", type="primary")

if run_button:
    if uploaded_file is None and not manual_path.strip():
        st.error("Upload an image or provide an image path.")
        st.stop()

    if uploaded_file is not None:
        image_path = save_uploaded_file(uploaded_file)
    else:
        image_path = Path(manual_path).expanduser()

    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        st.stop()

    with st.spinner("Running inference..."):
        result, output_dir = run_inference(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            min_mask_area=min_mask_area,
            preprocess=preprocess,
            keep_best_per_fdi=keep_best_per_fdi,
            show_scores=show_scores,
            display_preprocessed=display_preprocessed,
        )

    if result.returncode != 0:
        st.error("Inference failed.")
        st.code(result.stderr)
        st.stop()

    outputs = find_outputs(output_dir)

    st.success("Inference completed.")
    st.write(f"Output directory: `{output_dir}`")

    if outputs["png"]:
        st.subheader("Prediction")
        render_prediction_image(outputs["png"], display_mode, preview_width)
        st.caption("Use 'Fixed width' for normal screens. Use 'Scrollable original' if you want to inspect the full-resolution output.")

    if outputs["csv"]:
        st.subheader("Predicted teeth table")
        df = pd.read_csv(outputs["csv"])
        st.dataframe(df, use_container_width=True)

    if outputs["report"]:
        st.subheader("Report")
        report_text = outputs["report"].read_text()
        st.markdown(report_text)

    st.subheader("Downloads")

    zip_path = create_output_zip(output_dir)

    download_cols = st.columns(5)

    for col, label, path in zip(
        download_cols,
        ["PNG", "JSON", "CSV", "Markdown report", "Full ZIP"],
        [outputs["png"], outputs["json"], outputs["csv"], outputs["report"], zip_path],
    ):
        with col:
            if path and path.exists():
                st.download_button(
                    label=f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                )
