from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    python = sys.executable
    steps = []
    if not args.skip_grobid:
        steps.append([python, "scripts/pipeline/01_grobid_pdf_to_tei.py"])
    steps.extend(
        [
            [python, "scripts/pipeline/02_tei_to_local_s2orc.py"],
            [python, "scripts/qa/check_s2orc_quality.py"],
            [python, "scripts/pipeline/03_enrich_metadata.py"],
            [python, "scripts/pipeline/04_pes2o_filter_clean.py", "--strict" if args.strict else "--no-strict"],
            [python, "scripts/qa/check_pes2o_quality.py"],
            [python, "scripts/pipeline/05_openscholar_passages.py", "--strict" if args.strict else "--no-strict"],
            [python, "scripts/qa/check_passage_quality.py"],
            [python, "scripts/qa/pipeline_review.py"],
        ]
    )
    if args.rebuild_chroma:
        steps.append(
            [
                python,
                "scripts/pipeline/06_build_chroma.py",
                "--rebuild-chroma",
                "--batch-size",
                str(args.batch_size),
            ]
        )

    for command in steps:
        run_step(command)

    print("[pipeline] passages updated: data/passages/openscholar_passages.jsonl")
    print(f"[pipeline] chroma_rebuilt={args.rebuild_chroma}")
    print("[pipeline] reports: outputs/reports/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local PDF RAG preprocessing pipeline.")
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=False,
        help="Enable strict peS2o-compatible checks. Requires pycld3 and data/resources/unigram_freq.csv.",
    )
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--rebuild-chroma", action="store_true")
    parser.add_argument("--skip-grobid", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("[pipeline] running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
