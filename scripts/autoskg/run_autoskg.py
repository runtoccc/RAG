from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from .common import autoskg_config, ensure_deepseek_env, load_project_config, resolve_project_path
except ImportError:
    from common import autoskg_config, ensure_deepseek_env, load_project_config, resolve_project_path


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates" / "kg_project"


def main() -> None:
    args = parse_args()
    config = autoskg_config(load_project_config())
    ensure_deepseek_env(config)
    kg_root = resolve_project_path(args.kg_root or config["root_dir"])
    prepare_kg_project(kg_root, force=args.force_templates)

    env = build_env(config)
    if not args.skip_preprocess:
        run_step(
            "preprocess PDFs",
            [
                sys.executable,
                str(SCRIPT_DIR / "preprocess.py"),
                "--kg-root",
                str(kg_root),
                "--license-policy",
                args.license_policy or config["license_policy"],
            ],
            env,
        )

    if not args.skip_index:
        ensure_graphrag_available()
        patch_graphrag_logger()
        graphrag_cmd = find_graphrag_command()
        run_step(
            "build GraphRAG knowledge graph",
            [
                graphrag_cmd,
                "index",
                "--root",
                str(kg_root),
                "--config",
                str(kg_root / "settings.yaml"),
            ],
            env,
        )

    if not args.skip_postprocess:
        command = [sys.executable, str(SCRIPT_DIR / "postprocess.py"), "--kg-root", str(kg_root)]
        if args.no_fill_missing:
            command.append("--no-fill-missing")
        run_step("postprocess GraphRAG output", command, env)

    print("[autoskg] done")
    print(f"[autoskg] kg_root={kg_root}")
    print(f"[autoskg] output={kg_root / 'output'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local autoSKG-style GraphRAG pipeline.")
    parser.add_argument("--kg-root", default=None)
    parser.add_argument("--license-policy", choices=["allow_all", "cc_by"], default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--no-fill-missing", action="store_true")
    parser.add_argument("--force-templates", action="store_true")
    return parser.parse_args()


def prepare_kg_project(kg_root: Path, force: bool) -> None:
    kg_root.mkdir(parents=True, exist_ok=True)
    for dirname in ["input", "output", "cache", "logs"]:
        (kg_root / dirname).mkdir(parents=True, exist_ok=True)

    prompts_src = TEMPLATE_DIR / "prompts"
    prompts_dst = kg_root / "prompts"
    if force and prompts_dst.exists():
        shutil.rmtree(prompts_dst)
    if not prompts_dst.exists():
        shutil.copytree(prompts_src, prompts_dst)
    else:
        for prompt_path in prompts_src.glob("*.txt"):
            destination = prompts_dst / prompt_path.name
            if force or not destination.exists():
                shutil.copy2(prompt_path, destination)

    settings_src = TEMPLATE_DIR / "settings.yaml"
    settings_dst = kg_root / "settings.yaml"
    if force or not settings_dst.exists():
        shutil.copy2(settings_src, settings_dst)


def build_env(config: dict) -> dict[str, str]:
    env = os.environ.copy()
    api_key = env.get(config["llm_api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {config['llm_api_key_env']}")
    env.setdefault("DEEPSEEK_API_KEY", api_key)
    env.setdefault("DEEPSEEK_BASE_URL", config["llm_base_url"])
    env.setdefault("DEEPSEEK_MODEL", config["llm_model"])
    env.setdefault("OPENAI_API_KEY", api_key)
    env.setdefault("OPENAI_BASE_URL", config["llm_base_url"])
    env.setdefault("OPENAI_API_PROXY", config["llm_base_url"])
    env.setdefault("PYTHONHASHSEED", "12345")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def ensure_graphrag_available() -> None:
    try:
        import graphrag  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "GraphRAG is not installed. Run: pip install -r requirements.txt"
        ) from error


def patch_graphrag_logger() -> None:
    try:
        import graphrag
    except ModuleNotFoundError:
        return

    graphrag_dir = Path(graphrag.__file__).resolve().parent
    for factory_path in graphrag_dir.rglob("logger/factory.py"):
        text = factory_path.read_text(encoding="utf-8")
        if "GraphRAG Indexer" in text:
            factory_path.write_text(text.replace("GraphRAG Indexer", ""), encoding="utf-8")
        return


def find_graphrag_command() -> str:
    command = shutil.which("graphrag")
    if command:
        return command
    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    exe_path = scripts_dir / "graphrag.exe"
    if exe_path.exists():
        return str(exe_path)
    return "graphrag"


def run_step(label: str, command: list[str], env: dict[str, str]) -> None:
    print(f"[autoskg] {label}: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
