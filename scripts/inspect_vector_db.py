from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui_helpers import get_vector_db_status


def main():
    status = get_vector_db_status()
    serializable = {
        "path": str(status["path"]),
        "exists": status["exists"],
        "collection_name": status["collection_name"],
        "chunk_count": status["chunk_count"],
        "metadata_examples": status["metadata_examples"],
        "error": status["error"],
    }
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
