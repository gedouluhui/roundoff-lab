"""Command-line entry point for local matrix diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .analysis import analyze
from .demo import make_demo
from .io import load_configs, load_npy, npy_bytes, write_artifacts
from .report import render_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roundofflab", description=
        "Compare weight error and linear-layer output error. CPU fake quantization, no uploads.")
    parser.add_argument("--version", action="version", version="roundofflab 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="generate synthetic inputs and a self-contained report")
    demo.add_argument("--scenario", choices=("quiet-outlier", "active-outlier"), default="quiet-outlier")
    demo.add_argument("--seed", type=int, default=7)
    run = sub.add_parser("analyze", help="analyze local float32/64 .npy weight and input matrices")
    run.add_argument("--weights", type=Path, required=True, help="W[out, in]")
    run.add_argument("--inputs", type=Path, required=True, help="X[samples, in]")
    for command in (demo, run):
        command.add_argument("--out", type=Path, required=True, help="output directory (local files only)")
        command.add_argument("--configs", type=Path, help="explicit JSON array of quantization policies")
        command.add_argument("--title", help="report heading")
        command.add_argument("--force", action="store_true", help="replace existing output files")
        command.add_argument("--max-elements", type=int, default=4_000_000,
                             help="aggregate W/X element guard, not a peak-memory guarantee")
        command.add_argument("--max-output-elements", type=int, default=4_000_000)
    args = parser.parse_args(argv)
    try:
        configs = load_configs(args.configs) if args.configs else None
        input_paths = (args.configs,) if args.configs else ()
        if args.command == "demo":
            w, x = make_demo(args.seed, args.scenario)
            sample = f"synthetic / {args.scenario} (seed {args.seed})"
            title = args.title or ("When the outlier goes quiet" if args.scenario == "quiet-outlier"
                                   else "When the outlier wakes up")
        else:
            w, x = load_npy(args.weights), load_npy(args.inputs)
            input_paths += (args.weights, args.inputs)
            sample = "user-provided"
            title = args.title or "Your layer under quantization"
        report = analyze(w, x, configs, title=title, sample_label=sample,
                         max_elements=args.max_elements, max_output_elements=args.max_output_elements)
        payloads = {"report.json": (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
                    "report.html": render_report(report).encode("utf-8")}
        if args.command == "demo":
            payloads.update({"weights.npy": npy_bytes(w), "inputs.npy": npy_bytes(x)})
        write_artifacts(args.out, payloads, force=args.force, inputs=input_paths)
        print(f"Report: {args.out / 'report.html'}")
        print(f"JSON:   {args.out / 'report.json'}")
        print(f"Policies: {len(report['configurations'])}; strict ranking reversals: {len(report['ranking_reversals'])}")
        print("Single-layer evidence only; not a model-accuracy or performance benchmark.")
        return 0
    except (ValueError, TypeError, OSError, MemoryError, OverflowError) as exc:
        print(f"roundofflab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
