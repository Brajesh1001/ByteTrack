"""
Parse evaluation logs from all ByteTrack model variants and produce a
unified MOT15 comparison table in Markdown.

Usage:
    python tools/generate_mot15_comparison_table.py \
        --log-dir YOLOX_outputs \
        --output mot15_all_models_results.md
"""
import argparse
import os
import re
import json


MODEL_META = {
    "yolox_nano_mot15":  {"label": "ByteTrack-Nano",  "params": "0.90",  "flops": "3.99",   "test_size": "608x1088"},
    "yolox_tiny_mot15":  {"label": "ByteTrack-Tiny",  "params": "5.03",  "flops": "24.45",  "test_size": "608x1088"},
    "yolox_s_mot15":     {"label": "ByteTrack-S",     "params": "9.0",   "flops": "26.8",   "test_size": "608x1088"},
    "yolox_m_mot15":     {"label": "ByteTrack-M",     "params": "25.3",  "flops": "73.8",   "test_size": "800x1440"},
    "yolox_l_mot15":     {"label": "ByteTrack-L",     "params": "54.2",  "flops": "155.6",  "test_size": "800x1440"},
    "yolox_x_mot15":     {"label": "ByteTrack-X",     "params": "99.1",  "flops": "281.9",  "test_size": "800x1440"},
}

ORDERED_MODELS = [
    "yolox_nano_mot15",
    "yolox_tiny_mot15",
    "yolox_s_mot15",
    "yolox_m_mot15",
    "yolox_l_mot15",
    "yolox_x_mot15",
]

MOTCHALLENGE_HEADER_COLS = [
    "IDF1", "IDP", "IDR", "Rcll", "Prcn",
    "GT", "MT", "PT", "ML",
    "FP", "FN", "IDs", "FM",
    "MOTA", "MOTP",
]


def find_log_file(log_dir, model_name):
    candidate = os.path.join(log_dir, model_name, "val_log.txt")
    if os.path.isfile(candidate):
        return candidate
    candidate2 = os.path.join(log_dir, f"{model_name}.log")
    if os.path.isfile(candidate2):
        return candidate2
    return None


def parse_overall_row(log_text):
    """Extract the OVERALL row from the standard motchallenge metrics table."""
    results = {}

    for line in log_text.splitlines():
        if "OVERALL" not in line:
            continue
        cols = line.split()
        try:
            idx = cols.index("OVERALL")
        except ValueError:
            continue
        vals = cols[idx + 1:]
        if len(vals) < 15:
            continue
        for i, col_name in enumerate(MOTCHALLENGE_HEADER_COLS):
            results[col_name] = vals[i]
        break

    return results


def parse_per_sequence(log_text):
    """Extract per-sequence rows from the standard motchallenge table."""
    rows = []
    in_table = False
    for line in log_text.splitlines():
        stripped = line.strip()
        if "IDF1" in stripped and "MOTA" in stripped and "MOTP" in stripped:
            in_table = True
            continue
        if in_table:
            if stripped == "":
                break
            cols = stripped.split()
            if len(cols) >= 16:
                seq_name = cols[0]
                vals = cols[1:]
                row = {"sequence": seq_name}
                for i, col_name in enumerate(MOTCHALLENGE_HEADER_COLS):
                    if i < len(vals):
                        row[col_name] = vals[i]
                rows.append(row)
    return rows


def parse_timing(log_text):
    """Extract average forward/track/inference time from log."""
    timing = {}
    for line in log_text.splitlines():
        m = re.search(r"Average forward time:\s*([\d.]+)\s*ms", line)
        if m:
            timing["forward_ms"] = m.group(1)
        m = re.search(r"Average track time:\s*([\d.]+)\s*ms", line)
        if m:
            timing["track_ms"] = m.group(1)
        m = re.search(r"Average inference time:\s*([\d.]+)\s*ms", line)
        if m:
            timing["inference_ms"] = m.group(1)
    return timing


def build_markdown(all_results):
    lines = []
    lines.append("# ByteTrack All Models -- MOT15 Comparison Table\n")

    # ---- Main comparison table ----
    lines.append("## Overall Comparison\n")
    lines.append(
        "| Model | Params(M) | FLOPs(G) | Test Size | MOTA | IDF1 | MOTP "
        "| Rcll | Prcn | MT | ML | FP | FN | IDs | FM |"
    )
    lines.append(
        "|-------|-----------|----------|-----------|------|------|------"
        "|------|------|----|----|----|----|-----|-----|"
    )
    for model_key in ORDERED_MODELS:
        meta = MODEL_META[model_key]
        r = all_results.get(model_key, {}).get("overall", {})
        if not r:
            lines.append(
                f"| {meta['label']} | {meta['params']} | {meta['flops']} "
                f"| {meta['test_size']} | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |"
            )
        else:
            lines.append(
                f"| **{meta['label']}** | {meta['params']} | {meta['flops']} "
                f"| {meta['test_size']} "
                f"| **{r.get('MOTA','--')}** | **{r.get('IDF1','--')}** | {r.get('MOTP','--')} "
                f"| {r.get('Rcll','--')} | {r.get('Prcn','--')} "
                f"| {r.get('MT','--')} | {r.get('ML','--')} "
                f"| {r.get('FP','--')} | {r.get('FN','--')} "
                f"| {r.get('IDs','--')} | {r.get('FM','--')} |"
            )
    lines.append("")

    # ---- Timing table ----
    has_timing = any("timing" in v and v["timing"] for v in all_results.values())
    if has_timing:
        lines.append("## Inference Speed\n")
        lines.append("| Model | Forward (ms) | Track (ms) | Total (ms) | ~FPS |")
        lines.append("|-------|-------------|------------|------------|------|")
        for model_key in ORDERED_MODELS:
            meta = MODEL_META[model_key]
            t = all_results.get(model_key, {}).get("timing", {})
            if t:
                total = float(t.get("inference_ms", 0))
                fps = f"{1000.0/total:.1f}" if total > 0 else "--"
                lines.append(
                    f"| {meta['label']} | {t.get('forward_ms','--')} "
                    f"| {t.get('track_ms','--')} | {t.get('inference_ms','--')} | {fps} |"
                )
            else:
                lines.append(f"| {meta['label']} | -- | -- | -- | -- |")
        lines.append("")

    # ---- Per-sequence detail for each model ----
    for model_key in ORDERED_MODELS:
        meta = MODEL_META[model_key]
        seqs = all_results.get(model_key, {}).get("sequences", [])
        if not seqs:
            continue
        lines.append(f"## {meta['label']} -- Per-Sequence Results\n")
        lines.append("| Sequence | MOTA | IDF1 | MOTP | Rcll | Prcn | MT | ML | FP | FN | IDs | FM |")
        lines.append("|----------|------|------|------|------|------|----|----|----|----|-----|-----|")
        for row in seqs:
            lines.append(
                f"| {row['sequence']} "
                f"| {row.get('MOTA','--')} | {row.get('IDF1','--')} | {row.get('MOTP','--')} "
                f"| {row.get('Rcll','--')} | {row.get('Prcn','--')} "
                f"| {row.get('MT','--')} | {row.get('ML','--')} "
                f"| {row.get('FP','--')} | {row.get('FN','--')} "
                f"| {row.get('IDs','--')} | {row.get('FM','--')} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="YOLOX_outputs",
                        help="Directory containing per-model output folders")
    parser.add_argument("--output", default="mot15_all_models_results.md",
                        help="Output markdown file")
    args = parser.parse_args()

    all_results = {}

    for model_key in ORDERED_MODELS:
        log_path = find_log_file(args.log_dir, model_key)
        if log_path is None:
            print(f"[SKIP] No log found for {model_key}")
            continue

        print(f"[PARSE] {model_key} <- {log_path}")
        with open(log_path) as f:
            log_text = f.read()

        overall = parse_overall_row(log_text)
        sequences = parse_per_sequence(log_text)
        timing = parse_timing(log_text)

        all_results[model_key] = {
            "overall": overall,
            "sequences": sequences,
            "timing": timing,
        }

    md = build_markdown(all_results)
    with open(args.output, "w") as f:
        f.write(md)

    print(f"\nResults written to {args.output}")
    print(md)


if __name__ == "__main__":
    main()
