from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()
    run_step("register papers", [script("01_register_papers.py")])
    run_step("extract entity candidates", [script("02_extract_entity_candidates.py")])
    run_step("extract relation candidates", [script("03_extract_relation_candidates.py")])
    run_step("build local KG", [script("04_build_local_kg.py")])

    hybrid_cmd = [
        script("05_hybrid_retrieve.py"),
        args.question,
        "--top-k",
        str(args.top_k),
        "--node-k",
        str(args.node_k),
        "--edge-k",
        str(args.edge_k),
        "--output",
        args.bundle_output,
    ]
    if args.skip_vector:
        hybrid_cmd.append("--skip-vector")
    run_step("hybrid retrieve", hybrid_cmd)

    run_step(
        "build evidence answer",
        [
            script("06_answer_with_evidence.py"),
            "--bundle",
            args.bundle_output,
            "--output",
            args.answer_output,
            "--max-claims",
            str(args.max_claims),
        ],
    )

    if args.make_viewer:
        run_step(
            "export KG viewer data",
            [
                script("07_export_kg_viewer_data.py"),
                "--bundle",
                args.bundle_output,
            ],
        )
        run_step("make KG viewer HTML", [script("08_make_kg_viewer.py")])

    print("[agent-pipeline] done")
    print(f"[agent-pipeline] bundle={args.bundle_output}")
    print(f"[agent-pipeline] answer={args.answer_output}")
    print("[agent-pipeline] viewer data: python scripts/agent/07_export_kg_viewer_data.py --bundle data/agent/evidence_bundle.json")
    print("[agent-pipeline] viewer html: python scripts/agent/08_make_kg_viewer.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local literature-agent layer over existing passages.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--node-k", type=int, default=5)
    parser.add_argument("--edge-k", type=int, default=6)
    parser.add_argument("--max-claims", type=int, default=4)
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--bundle-output", default="data/agent/evidence_bundle.json")
    parser.add_argument("--answer-output", default="data/agent/evidence_answer.json")
    parser.add_argument("--make-viewer", action="store_true")
    return parser.parse_args()


def script(name: str) -> str:
    return str(SCRIPT_DIR / name)


def run_step(label: str, args: list[str]) -> None:
    command = [sys.executable, *args]
    print(f"[agent-pipeline] {label}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
