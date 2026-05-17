from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from tqdm import tqdm

try:
    from .common import autoskg_config, load_project_config, resolve_project_path
except ImportError:
    from common import autoskg_config, load_project_config, resolve_project_path


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
METADATA_COLUMNS = [
    "file_name",
    "source_file",
    "title",
    "authors",
    "doi",
    "md5",
    "pdf_sha256",
    "publication_date",
    "keywords",
    "license_policy",
    "license_status",
    "license",
    "license_error",
    "processed_at",
]


def main() -> None:
    args = parse_args()
    config = autoskg_config(load_project_config())
    input_dir = resolve_project_path(args.input_dir or config["input_dir"])
    tei_dir = resolve_project_path(args.tei_dir or config["tei_dir"])
    kg_root = resolve_project_path(args.kg_root or config["root_dir"])
    grobid_url = args.grobid_url or config["grobid_url"]
    license_policy = args.license_policy or config["license_policy"]

    process_all(
        input_dir=input_dir,
        tei_dir=tei_dir,
        kg_root=kg_root,
        grobid_url=grobid_url,
        prefer_existing_tei=not args.no_prefer_existing_tei and config["prefer_existing_tei"],
        license_policy=license_policy,
        unpaywall_email=args.unpaywall_email or config["unpaywall_email"],
        force_grobid=args.force_grobid,
        force_text=args.force_text,
        timeout=args.timeout,
        retries=args.retries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="autoSKG-style PDF/TEI preprocessing for local GraphRAG.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--tei-dir", default=None)
    parser.add_argument("--kg-root", default=None)
    parser.add_argument("--grobid-url", default=None)
    parser.add_argument("--license-policy", choices=["allow_all", "cc_by"], default=None)
    parser.add_argument("--unpaywall-email", default=None)
    parser.add_argument("--no-prefer-existing-tei", action="store_true")
    parser.add_argument("--force-grobid", action="store_true")
    parser.add_argument("--force-text", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def process_all(
    input_dir: Path,
    tei_dir: Path,
    kg_root: Path,
    grobid_url: str,
    prefer_existing_tei: bool,
    license_policy: str,
    unpaywall_email: str,
    force_grobid: bool,
    force_text: bool,
    timeout: int,
    retries: int,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input PDF directory does not exist: {input_dir}")

    pdf_paths = sorted(input_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    if not pdf_paths:
        raise FileNotFoundError(f"Please put PDF files in {input_dir}")

    input_text_dir = kg_root / "input"
    output_dir = kg_root / "output"
    tei_dir.mkdir(parents=True, exist_ok=True)
    input_text_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prefer_existing_tei or force_grobid:
        ensure_grobid_alive(grobid_url, timeout=timeout)

    metadata_rows: list[dict[str, str | None]] = []
    print(f"[autoskg-preprocess] pdf_count={len(pdf_paths)} input_dir={input_dir}")
    print(f"[autoskg-preprocess] kg_input={input_text_dir}")
    print(f"[autoskg-preprocess] license_policy={license_policy}")

    for pdf_path in tqdm(pdf_paths, desc="autoskg-preprocess"):
        try:
            row = process_pdf(
                pdf_path=pdf_path,
                tei_dir=tei_dir,
                input_text_dir=input_text_dir,
                grobid_url=grobid_url,
                prefer_existing_tei=prefer_existing_tei,
                license_policy=license_policy,
                unpaywall_email=unpaywall_email,
                force_grobid=force_grobid,
                force_text=force_text,
                timeout=timeout,
                retries=retries,
            )
            if row:
                metadata_rows.append(row)
        except Exception as error:
            print(f"[autoskg-preprocess] failed {pdf_path.name}: {error}", file=sys.stderr)

    write_metadata(output_dir / "metadata.parquet", metadata_rows)
    print(f"[autoskg-preprocess] metadata_rows={len(metadata_rows)}")
    print("[autoskg-preprocess] done")


def process_pdf(
    pdf_path: Path,
    tei_dir: Path,
    input_text_dir: Path,
    grobid_url: str,
    prefer_existing_tei: bool,
    license_policy: str,
    unpaywall_email: str,
    force_grobid: bool,
    force_text: bool,
    timeout: int,
    retries: int,
) -> dict[str, str | None] | None:
    tei_path = tei_dir / f"{pdf_path.stem}.tei.xml"
    output_path = input_text_dir / f"{pdf_path.stem}.txt"

    if output_path.exists() and not force_text and not force_grobid:
        print(f"[autoskg-preprocess] skip existing text {output_path.name}")
        return None

    if tei_path.exists() and prefer_existing_tei and not force_grobid:
        xml_text = tei_path.read_text(encoding="utf-8")
    else:
        xml_text = parse_pdf_with_grobid(pdf_path, grobid_url, timeout=timeout, retries=retries)
        tei_path.write_text(xml_text, encoding="utf-8")

    metadata, text = extract_text(xml_text)
    if not metadata or not text:
        return None

    license_info = check_license(metadata.get("doi"), unpaywall_email, license_policy)
    if not license_info["is_allowed"]:
        print(f"[autoskg-preprocess] skip {pdf_path.name}: {license_info['error']}")
        return None

    output_path.write_text(text, encoding="utf-8")

    file_name = pdf_path.stem
    return {
        "file_name": file_name,
        "source_file": pdf_path.name,
        "title": metadata.get("title") or file_name,
        "authors": metadata.get("authors"),
        "doi": metadata.get("doi"),
        "md5": metadata.get("md5") or md5_file(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "publication_date": metadata.get("publication_date"),
        "keywords": metadata.get("keywords"),
        "license_policy": license_policy,
        "license_status": license_info["status"],
        "license": license_info.get("license"),
        "license_error": license_info.get("error"),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }


def ensure_grobid_alive(grobid_url: str, timeout: int) -> None:
    alive_url = f"{grobid_base_url(grobid_url)}/api/isalive"
    response = requests.get(alive_url, timeout=min(timeout, 30))
    if response.status_code != 200 or response.text.strip().lower() != "true":
        raise RuntimeError(f"GROBID is not alive at {alive_url}: HTTP {response.status_code} {response.text[:200]}")


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


def parse_pdf_with_grobid(pdf_path: Path, grobid_url: str, timeout: int, retries: int) -> str:
    last_error: Exception | None = None
    ensure_grobid_alive(grobid_url, timeout)
    for attempt in range(1, retries + 1):
        try:
            with pdf_path.open("rb") as file:
                response = requests.post(
                    grobid_url,
                    files={"input": (pdf_path.name, file, "application/pdf")},
                    data={"consolidateHeader": "1", "consolidateCitations": "0"},
                    timeout=timeout,
                )
            if response.status_code == 200 and response.text.strip().startswith("<"):
                return response.text
            raise RuntimeError(f"GROBID returned HTTP {response.status_code}: {response.text[:300]}")
        except (requests.Timeout, requests.ConnectionError, RuntimeError) as error:
            last_error = error
            if attempt == retries:
                break
            print(f"[autoskg-preprocess] retry {pdf_path.name} attempt={attempt}/{retries}: {error}")
            time.sleep(3)
    raise RuntimeError(f"Failed to parse {pdf_path.name}: {last_error}")


def extract_text(xml_text: str) -> tuple[dict[str, str | None] | None, str | None]:
    try:
        tree = ET.fromstring(xml_text)
    except ET.ParseError as error:
        print(f"[autoskg-preprocess] XML parse error: {error}", file=sys.stderr)
        return None, None
    metadata = extract_metadata(tree)
    text = extract_text_content(tree)
    return metadata, text


def extract_metadata(tree: ET.Element) -> dict[str, str | None]:
    title = first_text(tree, ".//tei:titleStmt/tei:title[@type='main']")
    if not title:
        title = first_text(tree, ".//tei:titleStmt/tei:title")

    authors = []
    for author in tree.findall(".//tei:analytic/tei:author", TEI_NS):
        pers_name = author.find("tei:persName", TEI_NS)
        if pers_name is None:
            continue
        forename = pers_name.find("tei:forename[@type='first']", TEI_NS)
        surname = pers_name.find("tei:surname", TEI_NS)
        author_name = f"{safe_text(forename)} {safe_text(surname)}".strip()
        if author_name:
            authors.append(author_name)

    keywords = [
        safe_text(term).strip()
        for term in tree.findall(".//tei:profileDesc/tei:textClass/tei:keywords/tei:term", TEI_NS)
        if safe_text(term).strip()
    ]

    return {
        "title": title,
        "authors": "; ".join(authors),
        "doi": first_text(tree, ".//tei:idno[@type='DOI']"),
        "md5": first_text(tree, ".//tei:idno[@type='MD5']"),
        "publication_date": first_text(tree, ".//tei:publicationStmt/tei:date[@type='published']"),
        "keywords": "; ".join(keywords),
    }


def extract_text_content(tree: ET.Element) -> str:
    abstract_element = tree.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    abstract_text = extract_recursive_text(abstract_element) if abstract_element is not None else ""

    body_element = tree.find(".//tei:text/tei:body", TEI_NS)
    body_parts = []
    if body_element is not None:
        for div in body_element.findall(".//tei:div", TEI_NS):
            div_text = extract_div_text(div)
            if div_text:
                body_parts.append(div_text)

    combined = f"Abstract:\n{abstract_text}\n\nBody:\n{' '.join(body_parts)}"
    combined = re.sub(r"\[.*?\]", "", combined)
    combined = re.sub(r"\n+", "\n", combined)
    return combined.strip()


def extract_div_text(div: ET.Element) -> str:
    texts: list[str] = []
    for elem in list(div):
        if elem.tag == f"{{{TEI_NS['tei']}}}head":
            value = safe_text(elem).strip()
            if value:
                texts.append("\n" + value)
        elif elem.tag == f"{{{TEI_NS['tei']}}}p":
            value = extract_paragraph_text(elem)
            if value:
                texts.append(value)
        if elem.tail and elem.tail.strip():
            texts.append(elem.tail.strip())
    return "\n".join(texts)


def extract_paragraph_text(paragraph: ET.Element) -> str:
    texts: list[str] = []
    for elem in paragraph.iter():
        if elem.tag == f"{{{TEI_NS['tei']}}}ref" and elem.attrib.get("type") == "figure":
            continue
        if elem.text:
            texts.append(elem.text)
        if elem.tail:
            texts.append(elem.tail)
    return " ".join(" ".join(texts).split())


def extract_recursive_text(element: ET.Element) -> str:
    texts: list[str] = []
    for elem in element.iter():
        if elem.tag == f"{{{TEI_NS['tei']}}}ref" and elem.attrib.get("type") == "figure":
            continue
        if elem.tag == f"{{{TEI_NS['tei']}}}head":
            texts.append("\n")
        if elem.text:
            texts.append(elem.text)
        if elem.tail:
            texts.append(elem.tail)
    return "".join(texts)


def check_license(doi: str | None, email: str, policy: str) -> dict[str, str | bool | None]:
    if policy == "allow_all":
        return {"is_allowed": True, "status": "not_checked", "license": None, "error": None}

    if not doi:
        return {"is_allowed": False, "status": "rejected", "license": None, "error": "DOI is None"}

    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return {"is_allowed": False, "status": "rejected", "license": None, "error": "Could not find DOI"}
        response.raise_for_status()
        data = response.json()
        license_value = ((data.get("best_oa_location") or {}).get("license") or "").lower()
        return {
            "is_allowed": license_value == "cc-by",
            "status": "accepted" if license_value == "cc-by" else "rejected",
            "license": license_value,
            "error": None if license_value == "cc-by" else f"Paper is licensed under {license_value or 'unknown'}",
        }
    except Exception as error:
        return {"is_allowed": False, "status": "rejected", "license": None, "error": str(error)}


def write_metadata(path: Path, rows: list[dict[str, str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    if path.exists():
        existing_df = pd.read_parquet(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    if not combined.empty and "file_name" in combined.columns:
        combined = combined.drop_duplicates(subset=["file_name"], keep="last")
    combined.to_parquet(path, index=False)


def first_text(root: ET.Element, xpath: str) -> str | None:
    return safe_text(root.find(xpath, TEI_NS)) or None


def safe_text(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
