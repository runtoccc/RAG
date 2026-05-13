from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_loader import load_dotenv
from query_data import build_llm
from rag_config import load_config


def main():
    load_dotenv()
    config = load_config()
    try:
        llm = build_llm(config)
    except RuntimeError as error:
        print(error)
        print("Create .env in the project root, or set the variable in PowerShell.")
        return

    response = llm.invoke("Reply with exactly: API_OK")
    print(response.strip())


if __name__ == "__main__":
    main()
