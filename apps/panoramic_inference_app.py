import base64
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def render_interactive_prediction_image(image_path, selected_row=None):
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    fig = go.Figure(go.Image(z=image_np))

    if selected_row is not None:
        x1 = float(selected_row["bbox_x1"])
        y1 = float(selected_row["bbox_y1"])
        x2 = float(selected_row["bbox_x2"])
        y2 = float(selected_row["bbox_y2"])
        fdi = int(selected_row["fdi"])

        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line=dict(color="lime", width=4),
            fillcolor="rgba(0,255,0,0.08)",
        )
        fig.add_annotation(
            x=x1,
            y=y1,
            text=f"FDI {fdi}",
            showarrow=False,
            font=dict(color="white", size=14),
            bgcolor="rgba(0,120,0,0.85)",
            bordercolor="lime",
            borderwidth=1,
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        height=720,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def prepare_results_dataframe(df):
    df = df.copy()
    df["fdi"] = df["fdi"].astype(int)
    df["tooth"] = df["fdi"].astype(str)
    df["quadrant"] = df["tooth"].str[0].map({
        "1": "Q1 upper right",
        "2": "Q2 upper left",
        "3": "Q3 lower left",
        "4": "Q4 lower right",
    })
    df["confidence_band"] = pd.cut(
        df["score"],
        bins=[0.0, 0.65, 0.85, 1.01],
        labels=["Low <0.65", "Medium 0.65-0.85", "High ≥0.85"],
        include_lowest=True,
        right=False,
    )
    return df


def render_result_charts(df):
    if df.empty:
        st.warning("No predictions available for charts.")
        return

    df = prepare_results_dataframe(df)

    st.subheader("Result charts")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Detected teeth", len(df))

    with metric_col2:
        st.metric("Mean confidence", f"{df['score'].mean():.2f}")

    with metric_col3:
        st.metric("Lowest confidence", f"{df['score'].min():.2f}")

    with metric_col4:
        low_count = int((df["score"] < 0.75).sum())
        st.metric("Teeth <0.75", low_count)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Confidence by FDI")
        score_df = df.sort_values("fdi")
        fig_score = go.Figure()
        fig_score.add_bar(
            x=score_df["tooth"],
            y=score_df["score"],
            text=score_df["score"].round(2),
            textposition="outside",
        )
        fig_score.update_layout(
            xaxis_title="FDI",
            yaxis_title="Confidence",
            yaxis_range=[0, 1.05],
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with chart_col2:
        st.markdown("#### Mask area by FDI")
        area_df = df.sort_values("fdi")
        fig_area = go.Figure()
        fig_area.add_bar(
            x=area_df["tooth"],
            y=area_df["mask_area_pixels"],
            text=area_df["mask_area_pixels"],
            textposition="outside",
        )
        fig_area.update_layout(
            xaxis_title="FDI",
            yaxis_title="Mask area (pixels)",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_area, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### Detected teeth by quadrant")
        quadrant_df = (
            df.groupby("quadrant", observed=False)
            .size()
            .rename("count")
            .reset_index()
        )
        fig_quadrant = go.Figure()
        fig_quadrant.add_bar(
            x=quadrant_df["quadrant"],
            y=quadrant_df["count"],
            text=quadrant_df["count"],
            textposition="outside",
        )
        fig_quadrant.update_layout(
            xaxis_title="Quadrant",
            yaxis_title="Count",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_quadrant, use_container_width=True)

    with chart_col4:
        st.markdown("#### Confidence distribution")
        confidence_df = (
            df.groupby("confidence_band", observed=False)
            .size()
            .rename("count")
            .reset_index()
        )
        fig_confidence = go.Figure()
        fig_confidence.add_bar(
            x=confidence_df["confidence_band"].astype(str),
            y=confidence_df["count"],
            text=confidence_df["count"],
            textposition="outside",
        )
        fig_confidence.update_layout(
            xaxis_title="Confidence band",
            yaxis_title="Count",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_confidence, use_container_width=True)


    expected_fdis = (
        list(range(11, 19))
        + list(range(21, 29))
        + list(range(31, 39))
        + list(range(41, 49))
    )
    detected_fdis = set(df["fdi"].astype(int).tolist())
    fdi_status_df = pd.DataFrame({
        "FDI": expected_fdis,
        "Status": [
            "Detected" if fdi in detected_fdis else "Missing"
            for fdi in expected_fdis
        ],
    })

    st.markdown("#### Detected vs missing FDI")
    status_counts = (
        fdi_status_df["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )
    fig_status = go.Figure()
    fig_status.add_bar(
        x=status_counts["Status"],
        y=status_counts["Count"],
        text=status_counts["Count"],
        textposition="outside",
    )
    fig_status.update_layout(
        xaxis_title="Status",
        yaxis_title="Number of FDI positions",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_status, use_container_width=True)

    missing_fdis = fdi_status_df[fdi_status_df["Status"] == "Missing"]["FDI"].tolist()
    if missing_fdis:
        st.caption("Missing FDI positions: " + ", ".join(str(x) for x in missing_fdis))
    else:
        st.caption("No missing FDI positions.")

    low_confidence = df[df["score"] < 0.75].sort_values("score")

    if not low_confidence.empty:
        st.markdown("#### Teeth requiring review")
        st.write("These predictions have confidence below 0.75.")
        st.dataframe(
            low_confidence[
                ["fdi", "score", "mask_area_pixels", "centroid_x", "centroid_y"]
            ],
            use_container_width=True,
        )


def get_selected_tooth_row(df):
    if df.empty:
        return None

    df = df.sort_values("fdi").copy()
    fdi_options = df["fdi"].astype(int).tolist()

    selected_fdi = st.selectbox(
        "Select FDI tooth for detailed review",
        fdi_options,
    )

    return df[df["fdi"].astype(int) == int(selected_fdi)].iloc[0]


def render_tooth_review_panel(row):
    if row is None:
        return

    fdi = int(row["fdi"])
    score = float(row["score"])
    area = int(row["mask_area_pixels"])
    quadrant_id = str(fdi)[0]
    quadrant_label = {
        "1": "Q1 upper right",
        "2": "Q2 upper left",
        "3": "Q3 lower left",
        "4": "Q4 lower right",
    }.get(quadrant_id, "Unknown")

    if score >= 0.85:
        confidence_band = "High"
    elif score >= 0.65:
        confidence_band = "Medium"
    else:
        confidence_band = "Low"

    review_flag = "Review recommended" if score < 0.75 else "No immediate review flag"

    st.subheader("Selected tooth")

    st.metric("FDI", fdi)
    st.metric("Confidence", f"{score:.3f}")
    st.write(f"**Confidence band:** {confidence_band}")
    st.write(f"**Review flag:** {review_flag}")
    st.write(f"**Quadrant:** {quadrant_label}")
    st.write(f"**Mask area:** {area:,} pixels")

    st.markdown("**Coordinates**")
    coord_df = pd.DataFrame(
        {
            "Metric": ["Centroid x", "Centroid y", "BBox x1", "BBox y1", "BBox x2", "BBox y2"],
            "Value": [
                round(float(row["centroid_x"]), 1),
                round(float(row["centroid_y"]), 1),
                round(float(row["bbox_x1"]), 1),
                round(float(row["bbox_y1"]), 1),
                round(float(row["bbox_x2"]), 1),
                round(float(row["bbox_y2"]), 1),
            ],
        }
    )
    st.dataframe(coord_df, hide_index=True, use_container_width=True)

    mask_path = Path(str(row["mask_path"]))
    if mask_path.exists():
        st.image(str(mask_path), caption=f"Binary mask for FDI {fdi}", use_container_width=True)
    else:
        st.info("Mask file not found for preview.")


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
        ["Interactive zoom/pan", "Fixed width", "Fit page width", "Scrollable original"],
        index=0,
    )

with display_col2:
    preview_width = st.slider("Preview width (px)", 400, 1600, 850, 50)

viewer_content = st.radio(
    "Viewer content",
    ["Original image + selected FDI highlight", "Full segmentation overlay"],
    horizontal=True,
)

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

    st.session_state["last_output_dir"] = str(output_dir)

if "last_output_dir" in st.session_state:
    output_dir = Path(st.session_state["last_output_dir"])
    outputs = find_outputs(output_dir)

    st.success("Inference completed.")
    st.write(f"Output directory: `{output_dir}`")

    df = pd.read_csv(outputs["csv"]) if outputs["csv"] else pd.DataFrame()
    selected_row = None
    source_image_for_viewer = outputs["png"]

    if outputs["json"]:
        result_json = json.loads(outputs["json"].read_text())
        source_image_for_viewer = Path(result_json["image"])

    result_left, result_right = st.columns([3, 1])

    with result_right:
        if not df.empty:
            st.subheader("Tooth selection")
            selected_row = get_selected_tooth_row(df)
            render_tooth_review_panel(selected_row)

    with result_left:
        if outputs["png"]:
            st.subheader("Prediction")
            if display_mode == "Interactive zoom/pan":
                if viewer_content == "Full segmentation overlay":
                    render_interactive_prediction_image(outputs["png"], selected_row=None)
                    st.caption("Use mouse wheel to zoom, drag to pan, and double-click to reset the view. Showing all segmentation masks.")
                else:
                    render_interactive_prediction_image(source_image_for_viewer, selected_row=selected_row)
                    st.caption("Use mouse wheel to zoom, drag to pan, and double-click to reset the view. The selected FDI is highlighted in green.")
            else:
                render_prediction_image(outputs["png"], display_mode, preview_width)
                st.caption("Use 'Fixed width' for normal screens. Use 'Scrollable original' if you want to inspect the full-resolution output.")

    if not df.empty:
        st.subheader("Predicted teeth table")
        st.dataframe(df, use_container_width=True)
        render_result_charts(df)

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
