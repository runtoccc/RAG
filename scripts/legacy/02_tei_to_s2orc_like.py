from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tei_paths = sorted(input_dir.glob("*.tei.xml"), key=lambda path: path.name.lower())
    print(f"[tei2json] input_dir={input_dir}")
    print(f"[tei2json] tei_count={len(tei_paths)}")
    print(f"[tei2json] output={output_path}")

    with output_path.open("w", encoding="utf-8") as output_file:
        for tei_path in tei_paths:
            record = parse_tei_file(tei_path)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[ok] {tei_path.name} body_text={len(record['body_text'])} "
                f"bib_entries={len(record['bib_entries'])} status={record['parse_status']}"
            )

    print("[done] TEI XML converted to S2ORC-like JSONL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GROBID TEI XML to S2ORC-like JSONL.")
    parser.add_argument("--input-dir", default="data/interim/tei")
    parser.add_argument("--output", default="data/structured/s2orc_like.jsonl")
    return parser.parse_args()


def parse_tei_file(tei_path: Path) -> dict:
    source_file = source_file_from_tei_path(tei_path)
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError as error:
        paper_id = paper_id_from_source_file(source_file)
        return {
            "paper_id": paper_id,
            "source_file": source_file,
            "metadata": {"title": "", "authors": [], "year": None, "doi": None},
            "title": "",
            "abstract": "",
            "abstract_paragraphs": [],
            "body_text": [],
            "sections": [],
            "bib_entries": {},
            "ref_entries": {},
            "references": [],
            "parse_status": "xml_error",
            "quality_flags": [f"xml_error:{error}"],
        }

    title = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:titleStmt/tei:title")
    abstract_paragraphs = parse_abstract_paragraphs(root)
    abstract = "\n\n".join(paragraph["text"] for paragraph in abstract_paragraphs)
    authors = parse_authors(root)
    doi = extract_doi(root)
    year = extract_year(root)
    body_text, sections = parse_body(root)
    bib_entries = parse_bib_entries(root)
    references = [entry["text"] for entry in bib_entries.values() if entry.get("text")]
    quality_flags = build_quality_flags(
        title=title,
        abstract=abstract,
        body_text=body_text,
        doi=doi,
        year=year,
    )
    parse_status = determine_parse_status(quality_flags)
    paper_id = paper_id_from_doi(doi) if doi else paper_id_from_source_file(source_file)

    return {
        "paper_id": paper_id,
        "source_file": source_file,
        "metadata": {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
        },
        "title": title,
        "abstract": abstract,
        "abstract_paragraphs": abstract_paragraphs,
        "body_text": body_text,
        "sections": sections,
        "bib_entries": bib_entries,
        "ref_entries": {},
        "references": references,
        "parse_status": parse_status,
        "quality_flags": quality_flags,
    }


def source_file_from_tei_path(tei_path: Path) -> str:
    name = tei_path.name
    if name.endswith(".tei.xml"):
        return name[: -len(".tei.xml")] + ".pdf"
    return tei_path.with_suffix(".pdf").name


def parse_abstract_paragraphs(root: ET.Element) -> list[dict]:
    paragraphs = [
        normalize_space(element_text(element))
        for element in root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS)
    ]
    if not any(paragraphs):
        abstract = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        text = normalize_space(element_text(abstract)) if abstract is not None else ""
        paragraphs = [text] if text else []
    return [
        {
            "text": paragraph,
            "section": "Abstract",
            "cite_spans": [],
            "ref_spans": [],
        }
        for paragraph in paragraphs
        if paragraph
    ]


def parse_authors(root: ET.Element) -> list[str]:
    authors = []
    for author in root.findall(".//tei:teiHeader//tei:titleStmt/tei:author", TEI_NS):
        text = parse_author_name(author)
        if text:
            authors.append(text)
    return unique_keep_order(authors)


def parse_author_name(author: ET.Element) -> str:
    pers_name = author.find(".//tei:persName", TEI_NS)
    if pers_name is not None:
        forenames = [
            normalize_space(element_text(element))
            for element in pers_name.findall("tei:forename", TEI_NS)
        ]
        surname = normalize_space(element_text(pers_name.find("tei:surname", TEI_NS)))
        name = " ".join(part for part in [*forenames, surname] if part)
        if name:
            return name
    return normalize_space(element_text(author))


def extract_doi(root: ET.Element) -> str | None:
    for idno in root.findall(".//tei:idno", TEI_NS):
        id_type = (idno.attrib.get("type") or "").lower()
        text = normalize_doi(element_text(idno))
        if id_type == "doi" and text:
            return text
    match = DOI_RE.search(element_text(root))
    return normalize_doi(match.group(0)) if match else None


def normalize_doi(value: str | None) -> str | None:
    text = normalize_space(value or "").strip(". ")
    if not text:
        return None
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    return text.lower().rstrip(".,;")


def extract_year(root: ET.Element) -> int | None:
    date_paths = [
        ".//tei:teiHeader//tei:publicationStmt//tei:date",
        ".//tei:teiHeader//tei:sourceDesc//tei:date",
        ".//tei:teiHeader//tei:date",
        ".//tei:imprint//tei:date",
    ]
    for path in date_paths:
        for date in root.findall(path, TEI_NS):
            year = year_from_date_element(date)
            if year:
                return year
    return None


def year_from_date_element(date: ET.Element) -> int | None:
    values = [
        date.attrib.get("when"),
        date.attrib.get("from"),
        date.attrib.get("notBefore"),
        date.attrib.get("to"),
        date.attrib.get("notAfter"),
        element_text(date),
    ]
    for value in values:
        match = YEAR_RE.search(value or "")
        if match:
            return int(match.group(0))
    return None


def parse_body(root: ET.Element) -> tuple[list[dict], list[dict]]:
    body_text: list[dict] = []
    sections: list[dict] = []
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return body_text, sections

    direct_paragraphs = direct_child_texts(body, "p")
    if direct_paragraphs:
        sections.append(
            {"section_title": "Unknown", "sec_num": None, "paragraphs": direct_paragraphs}
        )
        for paragraph in direct_paragraphs:
            body_text.append(make_body_paragraph(paragraph, "Unknown", None))

    for div in body.findall("tei:div", TEI_NS):
        collect_div_body(div, inherited_title="Unknown", body_text=body_text, sections=sections)
    return body_text, sections


def collect_div_body(
    div: ET.Element,
    inherited_title: str,
    body_text: list[dict],
    sections: list[dict],
) -> None:
    heading = direct_child_text(div, "head") or inherited_title or "Unknown"
    sec_num = div.attrib.get("n")
    paragraphs = direct_child_texts(div, "p")
    if paragraphs:
        sections.append(
            {"section_title": heading, "sec_num": sec_num, "paragraphs": paragraphs}
        )
        for paragraph in paragraphs:
            body_text.append(make_body_paragraph(paragraph, heading, sec_num))

    for child_div in div.findall("tei:div", TEI_NS):
        collect_div_body(child_div, inherited_title=heading, body_text=body_text, sections=sections)


def make_body_paragraph(text: str, section: str, sec_num: str | None) -> dict:
    return {
        "text": text,
        "section": section,
        "sec_num": sec_num,
        "cite_spans": [],
        "ref_spans": [],
        "eq_spans": [],
    }


def parse_bib_entries(root: ET.Element) -> dict[str, dict]:
    entries = []
    entries.extend(root.findall(".//tei:text/tei:back//tei:listBibl//tei:biblStruct", TEI_NS))
    entries.extend(root.findall(".//tei:text/tei:back//tei:listBibl//tei:bibl", TEI_NS))

    bib_entries: dict[str, dict] = {}
    for index, element in enumerate(entries):
        key = f"BIBREF{index}"
        bib_entries[key] = parse_bib_entry(element)
    return bib_entries


def parse_bib_entry(element: ET.Element) -> dict:
    text = normalize_space(element_text(element))
    title = first_text(element, ".//tei:analytic/tei:title") or first_text(
        element, ".//tei:monogr/tei:title"
    )
    authors = []
    for author in element.findall(".//tei:analytic/tei:author", TEI_NS):
        name = parse_author_name(author)
        if name:
            authors.append(name)
    if not authors:
        for author in element.findall(".//tei:author", TEI_NS):
            name = parse_author_name(author)
            if name:
                authors.append(name)
    venue = first_text(element, ".//tei:monogr/tei:title")
    year = None
    for date in element.findall(".//tei:date", TEI_NS):
        year = year_from_date_element(date)
        if year:
            break
    return {
        "text": text or None,
        "title": title or None,
        "authors": unique_keep_order(authors),
        "year": year,
        "venue": venue or None,
    }


def build_quality_flags(
    title: str,
    abstract: str,
    body_text: list[dict],
    doi: str | None,
    year: int | None,
) -> list[str]:
    flags = []
    if not title:
        flags.append("missing_title")
    if not abstract:
        flags.append("missing_abstract")
    if not body_text:
        flags.append("missing_body")
    if not doi:
        flags.append("missing_doi")
    if not year:
        flags.append("missing_year")
    return flags


def determine_parse_status(quality_flags: list[str]) -> str:
    if "missing_body" in quality_flags:
        return "empty_body"
    if "missing_title" in quality_flags or "missing_abstract" in quality_flags:
        return "partial"
    return "ok"


def paper_id_from_doi(doi: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")
    return f"doi_{normalized}" if normalized else paper_id_from_source_file(doi)


def paper_id_from_source_file(source_file: str) -> str:
    stem = Path(source_file).stem
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:10]
    return f"local_{digest}"


def direct_child_texts(element: ET.Element, tag: str) -> list[str]:
    values = []
    for child in element.findall(f"tei:{tag}", TEI_NS):
        text = normalize_space(element_text(child))
        if text:
            values.append(text)
    return values


def direct_child_text(element: ET.Element, tag: str) -> str:
    child = element.find(f"tei:{tag}", TEI_NS)
    return normalize_space(element_text(child)) if child is not None else ""


def first_text(root: ET.Element, path: str) -> str:
    element = root.find(path, TEI_NS)
    return normalize_space(element_text(element)) if element is not None else ""


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(text for text in element.itertext() if text)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


if __name__ == "__main__":
    main()
