from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from pathlib import Path
import sys
import time
from urllib.parse import urlparse, urlunparse

import requests


DEFAULT_GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
DEFAULT_MANIFEST_CSV = "data/interim/manifests/pdf_parse_manifest.csv"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    failure_path = Path(args.failure_csv)
    manifest_path = Path(args.manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_grobid_alive(args.grobid_url, args.timeout)

    pdf_paths = sorted(input_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    print(f"[parse] input_dir={input_dir}")
    print(f"[parse] output_dir={output_dir}")
    print(f"[parse] pdf_count={len(pdf_paths)}")

    failures: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    parsed_count = 0
    skipped_count = 0

    for pdf_path in pdf_paths:
        paper_id = make_paper_id(pdf_path)
        tei_path = output_dir / f"{pdf_path.stem}.tei.xml"
        now = datetime.now().isoformat(timespec="seconds")
        manifest_row = {
            "paper_id": paper_id,
            "source_file": pdf_path.name,
            "tei_file": str(tei_path),
            "grobid_url": args.grobid_url,
            "consolidate_header": str(args.consolidate_header),
            "consolidate_citations": str(args.consolidate_citations),
            "pdf_sha256": sha256_file(pdf_path),
            "tei_sha256": sha256_file(tei_path) if tei_path.exists() else "",
            "parse_status": "",
            "error_type": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
        }

        if tei_path.exists() and not args.force:
            print(f"[skip] {pdf_path.name} -> {tei_path.name}")
            manifest_row["parse_status"] = "skipped"
            manifest_rows.append(manifest_row)
            skipped_count += 1
            continue

        try:
            tei_text = parse_pdf_with_grobid(
                pdf_path,
                args.grobid_url,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                consolidate_header=args.consolidate_header,
                consolidate_citations=args.consolidate_citations,
            )
            tei_path.write_text(tei_text, encoding="utf-8")
            manifest_row["tei_sha256"] = sha256_file(tei_path)
            manifest_row["parse_status"] = "parsed"
            parsed_count += 1
            print(f"[ok] {pdf_path.name} -> {tei_path.name}")
        except Exception as error:
            message = str(error).replace("\n", " ")
            manifest_row["parse_status"] = "failed"
            manifest_row["error_type"] = error.__class__.__name__
            manifest_row["error"] = message
            failures.append(
                {
                    "timestamp": now,
                    "source_file": pdf_path.name,
                    "error_type": error.__class__.__name__,
                    "error": message,
                }
            )
            print(f"[fail] {pdf_path.name}: {message}", file=sys.stderr)
        manifest_rows.append(manifest_row)

    write_failures(failure_path, failures)
    write_manifest(manifest_path, manifest_rows)
    print(
        f"[done] parsed={parsed_count} skipped={skipped_count} failed={len(failures)} "
        f"failure_csv={failure_path} manifest_csv={manifest_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse local PDFs into GROBID TEI XML.")
    parser.add_argument("--input-dir", default="data/papers")
    parser.add_argument("--output-dir", default="data/interim/tei")
    parser.add_argument("--grobid-url", default=DEFAULT_GROBID_URL)
    parser.add_argument("--failure-csv", default="outputs/reports/parse_failures.csv")
    parser.add_argument("--manifest-csv", default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=int, default=3)
    parser.add_argument("--consolidate-header", type=int, choices=[0, 1], default=1)
    parser.add_argument("--consolidate-citations", type=int, choices=[0, 1], default=0)
    parser.add_argument("--force", action="store_true", help="Re-parse existing TEI files.")
    return parser.parse_args()


def ensure_grobid_alive(grobid_url: str, timeout: int) -> None:
    alive_url = f"{grobid_base_url(grobid_url)}/api/isalive"
    try:
        response = requests.get(alive_url, timeout=min(timeout, 30))
    except requests.RequestException as error:
        raise SystemExit(
            f"GROBID is not reachable at {alive_url}. Start local GROBID first. "
            f"Original error: {error}"
        ) from error

    if response.status_code != 200 or response.text.strip().lower() != "true":
        raise SystemExit(
            f"GROBID is not alive at {alive_url}. "
            f"HTTP {response.status_code}: {response.text[:300]}"
        )


def grobid_base_url(grobid_url: str) -> str:
    parsed = urlparse(grobid_url)
    path = parsed.path.rstrip("/")
    if "/api/" in path:
        base_path = path.split("/api/", 1)[0]
    elif path.endswith("/api"):
        base_path = path[: -len("/api")]
    else:
        base_path = ""
    return urlunparse((parsed.scheme, parsed.netloc, base_path, "", "", "")).rstrip("/")


def parse_pdf_with_grobid(
    pdf_path: Path,
    grobid_url: str,
    timeout: int,
    retries: int,
    sleep_seconds: int,
    consolidate_header: int,
    consolidate_citations: int,
) -> str:
    last_error: Exception | None = None
    retry_status_codes = {500, 503}

    for attempt in range(1, retries + 1):
        try:
            with pdf_path.open("rb") as file:
                response = requests.post(
                    grobid_url,
                    files={"input": (pdf_path.name, file, "application/pdf")},
                    data={
                        "consolidateHeader": str(consolidate_header),
                        "consolidateCitations": str(consolidate_citations),
                    },
                    timeout=timeout,
                )
            if response.status_code == 200:
                text = response.text.strip()
                if not text.startswith("<"):
                    raise RuntimeError("GROBID response does not look like XML")
                return text

            error = RuntimeError(
                f"GROBID returned HTTP {response.status_code}: {response.text[:300]}"
            )
            if response.status_code not in retry_status_codes or attempt == retries:
                raise error
            last_error = error
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = error
            if attempt == retries:
                raise

        print(
            f"[retry] {pdf_path.name} attempt={attempt}/{retries} "
            f"error={last_error}",
            file=sys.stderr,
        )
        time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to parse {pdf_path.name}: {last_error}")


def make_paper_id(pdf_path: Path) -> str:
    digest = hashlib.md5(pdf_path.stem.encode("utf-8")).hexdigest()[:10]
    return f"local_{digest}"


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    fieldnames = ["timestamp", "source_file", "error_type", "error"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "paper_id",
        "source_file",
        "tei_file",
        "grobid_url",
        "consolidate_header",
        "consolidate_citations",
        "pdf_sha256",
        "tei_sha256",
        "parse_status",
        "error_type",
        "error",
        "created_at",
        "updated_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
