import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_coco_eval(path):
    path = Path(path)
    if not path.exists():
        return None

    lines = path.read_text(errors="ignore").splitlines()
    ap_lines = [x for x in lines if "Average Precision" in x]

    values = []
    for line in ap_lines:
        match = re.search(r"=\s*(-?\d+\.\d+)", line)
        if match:
            values.append(float(match.group(1)))

    if len(values) < 12:
        return None

    return {
        "bbox_AP": values[0],
        "bbox_AP50": values[1],
        "bbox_AP75": values[2],
        "mask_AP": values[6],
        "mask_AP50": values[7],
        "mask_AP75": values[8],
    }


def make_plots(epoch_df, output_dir, run_name):
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if epoch_df["mean_loss"].notna().any():
        plt.figure(figsize=(9, 5))
        plt.plot(epoch_df["epoch"], epoch_df["mean_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Mean training loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{run_name}_training_loss.png", dpi=200)
        plt.close()

    if "val_mask_AP" in epoch_df.columns and epoch_df["val_mask_AP"].notna().any():
        val_df = epoch_df.dropna(subset=["val_mask_AP"])
        plt.figure(figsize=(9, 5))
        plt.plot(val_df["epoch"], val_df["val_mask_AP"])
        plt.xlabel("Epoch")
        plt.ylabel("Validation mask AP")
        plt.title("Validation mask AP")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{run_name}_val_mask_ap.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a local Mask R-CNN training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True, help="Run name, for example maskrcnn_b8_lr0002_valap_20260625_210821.")
    parser.add_argument("--test-eval", default=None, help="Optional COCO test evaluation txt file.")
    parser.add_argument("--per-tooth-csv", default=None, help="Optional per-tooth AP CSV.")
    parser.add_argument("--tooth-errors-csv", default=None, help="Optional tooth error cases CSV.")
    parser.add_argument("--output-dir", default="outputs/reports/run_analysis", help="Output directory.")
    args = parser.parse_args()

    run_name = args.run_name

    train_csv = Path("outputs/logs") / f"{run_name}_train_log.csv"
    terminal_log = Path("outputs/logs") / f"{run_name}_terminal.log"
    best_info_json = Path("outputs/checkpoints") / run_name / "best_model_info.json"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing training CSV: {train_csv}")

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    epoch_df = df[df["step"].astype(str) == "epoch_end"].copy()

    for col in epoch_df.columns:
        if col != "step":
            epoch_df[col] = pd.to_numeric(epoch_df[col], errors="coerce")

    final_epoch = int(epoch_df["epoch"].max())
    final_loss = float(epoch_df["mean_loss"].dropna().iloc[-1])

    min_loss_row = epoch_df.loc[epoch_df["mean_loss"].idxmin()]
    min_loss_epoch = int(min_loss_row["epoch"])
    min_loss = float(min_loss_row["mean_loss"])

    best_info = {}
    if best_info_json.exists():
        best_info = json.loads(best_info_json.read_text())

    log_text = terminal_log.read_text(errors="ignore") if terminal_log.exists() else ""
    early_stopped = "Early stopping triggered" in log_text

    lines = []
    lines.append(f"# Training analysis: `{run_name}`\n")

    lines.append("## Main summary\n")
    lines.append(f"- Final epoch reached: **{final_epoch}**")
    lines.append(f"- Final mean training loss: **{final_loss:.4f}**")
    lines.append(f"- Lowest mean training loss: **{min_loss:.4f}** at epoch **{min_loss_epoch}**")
    lines.append(f"- Early stopping triggered: **{'yes' if early_stopped else 'no'}**")

    if best_info:
        lines.append(f"- Best model epoch: **{best_info.get('best_epoch', 'NA')}**")
        if "best_val_mask_AP" in best_info:
            lines.append(f"- Best validation mask AP: **{best_info['best_val_mask_AP']:.4f}**")
        if "val_mask_AP50" in best_info:
            lines.append(f"- Best validation mask AP50: **{best_info['val_mask_AP50']:.4f}**")
        if "val_bbox_AP" in best_info:
            lines.append(f"- Best validation bbox AP: **{best_info['val_bbox_AP']:.4f}**")

    lines.append("\n## Validation trend\n")

    if "val_mask_AP" in epoch_df.columns and epoch_df["val_mask_AP"].notna().any():
        val_df = epoch_df.dropna(subset=["val_mask_AP"]).copy()
        best_val_row = val_df.loc[val_df["val_mask_AP"].idxmax()]
        last_val_row = val_df.iloc[-1]

        best_val_epoch = int(best_val_row["epoch"])
        best_val_ap = float(best_val_row["val_mask_AP"])
        last_val_epoch = int(last_val_row["epoch"])
        last_val_ap = float(last_val_row["val_mask_AP"])

        lines.append(f"- Best validation mask AP was **{best_val_ap:.4f}** at epoch **{best_val_epoch}**.")
        lines.append(f"- Last validation mask AP was **{last_val_ap:.4f}** at epoch **{last_val_epoch}**.")

        if last_val_ap < best_val_ap:
            lines.append("- Validation declined after the best epoch. Use `best_model.pth`, not the final epoch.")
        else:
            lines.append("- Validation did not decline at the end.")

        lines.append(f"- Training continued **{final_epoch - best_val_epoch} epochs** after the best model was found.")
    else:
        lines.append("- No validation AP found. This run was probably not selected by validation performance.")

    lines.append("\n## Instability check\n")

    loss_series = epoch_df[["epoch", "mean_loss"]].dropna()
    median_loss = loss_series["mean_loss"].median()
    spike_df = loss_series[loss_series["mean_loss"] > max(5 * median_loss, 5.0)]

    if len(spike_df) > 0:
        first_spike = spike_df.iloc[0]
        lines.append(
            f"- Training instability detected around epoch **{int(first_spike['epoch'])}** "
            f"with mean loss **{float(first_spike['mean_loss']):.4f}**."
        )
    else:
        lines.append("- No major loss explosion detected.")

    if args.test_eval:
        test_metrics = parse_coco_eval(args.test_eval)
        if test_metrics:
            lines.append("\n## Test-set performance\n")
            lines.append(f"- Test bbox AP: **{test_metrics['bbox_AP']:.3f}**")
            lines.append(f"- Test bbox AP50: **{test_metrics['bbox_AP50']:.3f}**")
            lines.append(f"- Test bbox AP75: **{test_metrics['bbox_AP75']:.3f}**")
            lines.append(f"- Test mask AP: **{test_metrics['mask_AP']:.3f}**")
            lines.append(f"- Test mask AP50: **{test_metrics['mask_AP50']:.3f}**")
            lines.append(f"- Test mask AP75: **{test_metrics['mask_AP75']:.3f}**")

    if args.per_tooth_csv and Path(args.per_tooth_csv).exists():
        tooth_df = pd.read_csv(args.per_tooth_csv)

        lines.append("\n## Per-tooth performance\n")

        worst = tooth_df.sort_values("mask_AP").head(5)
        best = tooth_df.sort_values("mask_AP", ascending=False).head(5)

        lines.append("Worst teeth by mask AP:")
        for _, row in worst.iterrows():
            lines.append(f"- FDI **{int(row['FDI'])}**: mask AP **{row['mask_AP']:.3f}**, mask AP50 **{row['mask_AP50']:.3f}**, n={int(row['n_gt'])}")

        lines.append("\nBest teeth by mask AP:")
        for _, row in best.iterrows():
            lines.append(f"- FDI **{int(row['FDI'])}**: mask AP **{row['mask_AP']:.3f}**, mask AP50 **{row['mask_AP50']:.3f}**, n={int(row['n_gt'])}")

    if args.tooth_errors_csv and Path(args.tooth_errors_csv).exists():
        err_df = pd.read_csv(args.tooth_errors_csv)

        lines.append("\n## Tooth-specific error cases\n")

        zero_iou = err_df[err_df["iou"] == 0].groupby("tooth").size().sort_values(ascending=False)

        if len(zero_iou) > 0:
            lines.append("IoU 0 cases among saved worst examples:")
            for tooth, count in zero_iou.items():
                lines.append(f"- FDI **{int(tooth)}**: {int(count)} cases")
        else:
            lines.append("- No IoU 0 cases found among saved worst examples.")

    lines.append("\n## Recommendation\n")
    lines.append("- Use the model selected by validation mask AP.")
    lines.append("- Compare this test result against the previous baseline.")
    lines.append("- Use per-tooth AP and worst-case visualizations to decide targeted improvements.")

    make_plots(epoch_df, output_dir, run_name)

    report_path = output_dir / f"{run_name}_analysis.md"
    report_path.write_text("\n".join(lines))

    print(f"Saved report: {report_path}")
    print(f"Saved figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
