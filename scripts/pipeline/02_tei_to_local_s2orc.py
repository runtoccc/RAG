from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    compat_output = Path(args.compat_output) if args.compat_output else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compat_output:
        compat_output.parent.mkdir(parents=True, exist_ok=True)

    tei_paths = sorted(input_dir.glob("*.tei.xml"), key=lambda path: path.name.lower())
    records = [parse_tei_file(path) for path in tei_paths]
    write_jsonl(output_path, records)
    if compat_output:
        write_jsonl(compat_output, records)

    print(f"[local-s2orc] input_dir={input_dir}")
    print(f"[local-s2orc] tei_count={len(tei_paths)}")
    print(f"[local-s2orc] output={output_path}")
    if compat_output:
        print(f"[local-s2orc] compat_output={compat_output}")
    print(f"[local-s2orc] records={len(records)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GROBID TEI XML to local S2ORC-aligned structured JSONL.")
    parser.add_argument("--input-dir", default="data/interim/tei")
    parser.add_argument("--output", default="data/structured/local_s2orc.jsonl")
    parser.add_argument("--compat-output", default="data/structured/s2orc_like.jsonl")
    return parser.parse_args()


def parse_tei_file(tei_path: Path) -> dict:
    source_file = source_file_from_tei_path(tei_path)
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError as error:
        return xml_error_record(source_file, error)

    title = first_text(root, ".//tei:teiHeader//tei:fileDesc//tei:titleStmt/tei:title")
    abstract_paragraphs = parse_abstract_paragraphs(root)
    abstract = "\n\n".join(paragraph["text"] for paragraph in abstract_paragraphs)
    authors = parse_authors(root)
    doi = extract_doi(root)
    year = extract_year(root)
    bib_entries = parse_bib_entries(root)
    ref_entries = parse_ref_entries(root)
    body_text, sections = parse_body(root)
    structural_flags, metadata_flags = build_quality_flags(
        title, abstract, body_text, doi, year, bib_entries, ref_entries
    )
    quality_flags = unique_keep_order(structural_flags + metadata_flags)
    parse_status = determine_parse_status(title, abstract, body_text, structural_flags)
    paper_id = paper_id_from_doi(doi) if doi else paper_id_from_source_file(source_file)
    cite_spans_count = sum(len(paragraph.get("cite_spans") or []) for paragraph in body_text)
    ref_spans_count = sum(len(paragraph.get("ref_spans") or []) for paragraph in body_text)

    return {
        "paper_id": paper_id,
        "source_file": source_file,
        "metadata": {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "metadata_source": "grobid",
            "s2orc_alignment_note": "s2orc_compatible_local_grobid_object",
            "citation_spans_status": "tei_bibr_spans" if cite_spans_count else "none_found",
            "resolved_citations_status": "not_semantic_scholar_resolved"
            if cite_spans_count
            else "none_found",
            "ref_entries_status": "tei_ref_entries" if ref_entries else "none_found",
            "ref_spans_status": "tei_ref_spans" if ref_spans_count else "none_found",
        },
        "title": title,
        "abstract": abstract,
        "abstract_paragraphs": abstract_paragraphs,
        "body_text": body_text,
        "sections": sections,
        "bib_entries": bib_entries,
        "ref_entries": ref_entries,
        "parse_status": parse_status,
        "structural_flags": structural_flags,
        "metadata_flags": metadata_flags,
        "quality_flags": quality_flags,
    }


def parse_abstract_paragraphs(root: ET.Element) -> list[dict]:
    paragraphs = root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS)
    if not paragraphs:
        abstract = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
        text = normalize_space(element_text(abstract)) if abstract is not None else ""
        return [make_abstract_paragraph(text)] if text else []
    return [
        make_abstract_paragraph(normalize_space(element_text(paragraph)))
        for paragraph in paragraphs
        if normalize_space(element_text(paragraph))
    ]


def make_abstract_paragraph(text: str) -> dict:
    return {"text": text, "section": "Abstract", "cite_spans": parse_cite_spans_from_text(text)}


def parse_body(root: ET.Element) -> tuple[list[dict], list[dict]]:
    body_text: list[dict] = []
    sections: list[dict] = []
    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return body_text, sections

    for paragraph in direct_child_elements(body, "p"):
        text = normalize_space(element_text(paragraph))
        if text:
            body_text.append(make_body_paragraph(paragraph, "Unknown", None))
    if body_text:
        sections.append(
            {
                "section_title": "Unknown",
                "sec_num": None,
                "paragraphs": [paragraph["text"] for paragraph in body_text],
            }
        )

    for div in body.findall("tei:div", TEI_NS):
        collect_div_body(div, "Unknown", body_text, sections)
    return body_text, sections


def collect_div_body(
    div: ET.Element,
    inherited_title: str,
    body_text: list[dict],
    sections: list[dict],
) -> None:
    heading = direct_child_text(div, "head") or inherited_title or "Unknown"
    sec_num = div.attrib.get("n")
    paragraph_items = []
    for paragraph in direct_child_elements(div, "p"):
        item = make_body_paragraph(paragraph, heading, sec_num)
        if item["text"]:
            body_text.append(item)
            paragraph_items.append(item["text"])
    if paragraph_items:
        sections.append({"section_title": heading, "sec_num": sec_num, "paragraphs": paragraph_items})
    for child_div in div.findall("tei:div", TEI_NS):
        collect_div_body(child_div, heading, body_text, sections)


def make_body_paragraph(paragraph: ET.Element, section: str, sec_num: str | None) -> dict:
    text, cite_spans, ref_spans, eq_spans = text_and_spans(paragraph)
    return {
        "text": text,
        "section": section,
        "sec_num": sec_num,
        "cite_spans": cite_spans,
        "ref_spans": ref_spans,
        "eq_spans": eq_spans,
    }


def text_and_spans(element: ET.Element) -> tuple[str, list[dict], list[dict], list[dict]]:
    text_parts: list[str] = []
    cite_spans: list[dict] = []
    ref_spans: list[dict] = []
    eq_spans: list[dict] = []

    def append_text(value: str | None) -> None:
        if value:
            text_parts.append(value)

    def current_text() -> str:
        return "".join(text_parts)

    def walk(node: ET.Element) -> None:
        append_text(node.text)
        for child in list(node):
            child_type = (child.attrib.get("type") or "").lower()
            if strip_namespace(child.tag) == "ref" and child_type == "bibr":
                start = len(current_text())
                append_text(element_text(child))
                end = len(current_text())
                target = child.attrib.get("target") or ""
                cite_spans.append(
                    {
                        "start": start,
                        "end": end,
                        "text": current_text()[start:end],
                        "ref_id": target.lstrip("#") or None,
                    }
                )
            elif strip_namespace(child.tag) == "ref" and child_type in {"figure", "table", "formula"}:
                start = len(current_text())
                append_text(element_text(child))
                end = len(current_text())
                target = child.attrib.get("target") or ""
                span = {
                    "start": start,
                    "end": end,
                    "text": current_text()[start:end],
                    "ref_id": target.lstrip("#") or None,
                    "ref_type": child_type,
                }
                ref_spans.append(span)
                if child_type == "formula":
                    eq_spans.append(dict(span))
            else:
                walk(child)
            append_text(child.tail)

    walk(element)
    text = normalize_space(current_text())
    return (
        text,
        normalize_span_offsets(text, cite_spans),
        normalize_span_offsets(text, ref_spans),
        normalize_span_offsets(text, eq_spans),
    )


def normalize_span_offsets(text: str, spans: list[dict]) -> list[dict]:
    normalized = []
    for span in spans:
        span_text = normalize_space(span.get("text") or "")
        if not span_text:
            continue
        start = text.find(span_text)
        normalized.append(
            {
                "start": start if start >= 0 else None,
                "end": start + len(span_text) if start >= 0 else None,
                "text": span_text,
                "ref_id": span.get("ref_id"),
                **({"ref_type": span.get("ref_type")} if span.get("ref_type") else {}),
            }
        )
    return normalized


def parse_cite_spans_from_text(_: str) -> list[dict]:
    return []


def parse_bib_entries(root: ET.Element) -> dict[str, dict]:
    entries = []
    entries.extend(root.findall(".//tei:text/tei:back//tei:listBibl//tei:biblStruct", TEI_NS))
    entries.extend(root.findall(".//tei:text/tei:back//tei:listBibl//tei:bibl", TEI_NS))
    bib_entries = {}
    for index, element in enumerate(entries):
        entry = parse_bib_entry(element, index)
        bib_entries[entry["ref_id"]] = entry
    return bib_entries


def parse_bib_entry(element: ET.Element, index: int) -> dict:
    ref_id = element.attrib.get("{http://www.w3.org/XML/1998/namespace}id") or f"BIBREF{index}"
    title = first_text(element, ".//tei:analytic/tei:title") or first_text(element, ".//tei:monogr/tei:title")
    venue = first_text(element, ".//tei:monogr/tei:title")
    doi = None
    for idno in element.findall(".//tei:idno", TEI_NS):
        if (idno.attrib.get("type") or "").lower() == "doi":
            doi = normalize_doi(element_text(idno))
            break
    year = None
    for date in element.findall(".//tei:date", TEI_NS):
        year = year_from_date_element(date)
        if year:
            break
    authors = [parse_author_name(author) for author in element.findall(".//tei:author", TEI_NS)]
    return {
        "ref_id": ref_id,
        "title": title or None,
        "authors": unique_keep_order([author for author in authors if author]),
        "year": year,
        "venue": venue or None,
        "doi": doi,
        "raw_text": normalize_space(element_text(element)) or None,
    }


def parse_ref_entries(root: ET.Element) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    counters = {"figure": 0, "table": 0, "formula": 0}
    for figure in root.findall(".//tei:figure", TEI_NS):
        fig_type = (figure.attrib.get("type") or "figure").lower()
        ref_type = "table" if fig_type == "table" else "figure"
        counters[ref_type] += 1
        ref_id = xml_id(figure) or f"{ref_type}_{counters[ref_type]}"
        entries[ref_id] = {
            "type": ref_type,
            "text": normalize_space(element_text(figure)),
            "caption": first_text(figure, ".//tei:figDesc") or first_text(figure, ".//tei:head"),
            "label": first_text(figure, ".//tei:label") or ref_id,
        }
    for table in root.findall(".//tei:table", TEI_NS):
        counters["table"] += 1
        ref_id = xml_id(table) or f"table_{counters['table']}"
        entries[ref_id] = {
            "type": "table",
            "text": normalize_space(element_text(table)),
            "caption": first_text(table, ".//tei:head"),
            "label": first_text(table, ".//tei:label") or ref_id,
        }
    for formula in root.findall(".//tei:formula", TEI_NS):
        counters["formula"] += 1
        ref_id = xml_id(formula) or f"formula_{counters['formula']}"
        entries[ref_id] = {
            "type": "formula",
            "text": normalize_space(element_text(formula)),
            "caption": "",
            "label": first_text(formula, ".//tei:label") or ref_id,
        }
    return entries


def build_quality_flags(
    title: str,
    abstract: str,
    body_text: list[dict],
    doi: str | None,
    year: int | None,
    bib_entries: dict,
    ref_entries: dict,
) -> tuple[list[str], list[str]]:
    structural_flags = []
    metadata_flags = []
    if not title:
        metadata_flags.append("missing_title")
    if not abstract:
        metadata_flags.append("missing_abstract")
    if not body_text:
        structural_flags.append("missing_body")
    if not doi:
        metadata_flags.append("missing_doi")
    if not year:
        metadata_flags.append("missing_year")
    cite_spans = [span for paragraph in body_text for span in paragraph.get("cite_spans") or []]
    ref_spans = [span for paragraph in body_text for span in paragraph.get("ref_spans") or []]
    if not cite_spans:
        structural_flags.append("no_cite_spans_found")
    unresolved = [span for span in cite_spans if span.get("ref_id") not in bib_entries]
    if unresolved:
        structural_flags.append("unresolved_local_citation_span")
    if not ref_entries:
        structural_flags.append("no_ref_entries_found")
    if not ref_spans:
        structural_flags.append("no_ref_spans_found")
    return unique_keep_order(structural_flags), unique_keep_order(metadata_flags)


def determine_parse_status(
    title: str,
    abstract: str,
    body_text: list[dict],
    structural_flags: list[str],
) -> str:
    if not body_text:
        return "empty_body"
    if not title or not abstract:
        return "partial_missing_metadata"
    weak_flags = {"no_cite_spans_found", "no_ref_entries_found", "no_ref_spans_found"}
    if weak_flags.issubset(set(structural_flags)):
        return "structurally_weak"
    return "ok"


def xml_error_record(source_file: str, error: Exception) -> dict:
    return {
        "paper_id": paper_id_from_source_file(source_file),
        "source_file": source_file,
        "metadata": {
            "title": "",
            "authors": [],
            "year": None,
            "doi": None,
            "metadata_source": "grobid",
            "s2orc_alignment_note": "xml_parse_failed",
            "citation_spans_status": "none_found",
            "resolved_citations_status": "none_found",
            "ref_entries_status": "none_found",
            "ref_spans_status": "none_found",
        },
        "title": "",
        "abstract": "",
        "abstract_paragraphs": [],
        "body_text": [],
        "sections": [],
        "bib_entries": {},
        "ref_entries": {},
        "parse_status": "xml_error",
        "structural_flags": ["xml_error", "missing_body"],
        "metadata_flags": ["missing_title", "missing_abstract", "missing_doi", "missing_year"],
        "quality_flags": ["xml_error", f"xml_error_detail:{error}"],
    }


def parse_authors(root: ET.Element) -> list[str]:
    authors = []
    for author in root.findall(".//tei:teiHeader//tei:titleStmt/tei:author", TEI_NS):
        name = parse_author_name(author)
        if name:
            authors.append(name)
    return unique_keep_order(authors)


def parse_author_name(author: ET.Element) -> str:
    pers_name = author.find(".//tei:persName", TEI_NS)
    if pers_name is not None:
        forenames = [normalize_space(element_text(el)) for el in pers_name.findall("tei:forename", TEI_NS)]
        surname = normalize_space(element_text(pers_name.find("tei:surname", TEI_NS)))
        name = " ".join(part for part in [*forenames, surname] if part)
        if name:
            return name
    return normalize_space(element_text(author))


def extract_doi(root: ET.Element) -> str | None:
    for idno in root.findall(".//tei:idno", TEI_NS):
        if (idno.attrib.get("type") or "").lower() == "doi":
            doi = normalize_doi(element_text(idno))
            if doi:
                return doi
    match = DOI_RE.search(element_text(root))
    return normalize_doi(match.group(0)) if match else None


def extract_year(root: ET.Element) -> int | None:
    for path in [
        ".//tei:teiHeader//tei:publicationStmt//tei:date",
        ".//tei:teiHeader//tei:sourceDesc//tei:date",
        ".//tei:teiHeader//tei:date",
        ".//tei:imprint//tei:date",
    ]:
        for date in root.findall(path, TEI_NS):
            year = year_from_date_element(date)
            if year:
                return year
    return None


def year_from_date_element(date: ET.Element) -> int | None:
    for value in [date.attrib.get("when"), date.attrib.get("from"), date.attrib.get("notBefore"), element_text(date)]:
        match = YEAR_RE.search(value or "")
        if match:
            return int(match.group(0))
    return None


def source_file_from_tei_path(tei_path: Path) -> str:
    return tei_path.name[: -len(".tei.xml")] + ".pdf" if tei_path.name.endswith(".tei.xml") else tei_path.with_suffix(".pdf").name


def paper_id_from_doi(doi: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")
    return f"doi_{normalized}" if normalized else paper_id_from_source_file(doi)


def paper_id_from_source_file(source_file: str) -> str:
    digest = hashlib.md5(Path(source_file).stem.encode("utf-8")).hexdigest()[:10]
    return f"local_{digest}"


def direct_child_elements(element: ET.Element, tag: str) -> list[ET.Element]:
    return list(element.findall(f"tei:{tag}", TEI_NS))


def direct_child_text(element: ET.Element, tag: str) -> str:
    child = element.find(f"tei:{tag}", TEI_NS)
    return normalize_space(element_text(child)) if child is not None else ""


def first_text(root: ET.Element, path: str) -> str:
    element = root.find(path, TEI_NS)
    return normalize_space(element_text(element)) if element is not None else ""


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext())


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_id(element: ET.Element) -> str | None:
    return element.attrib.get("{http://www.w3.org/XML/1998/namespace}id")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_doi(value: str | None) -> str | None:
    text = normalize_space(value or "").lower()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    return text.rstrip(".,;") or None


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        key = value.lower()
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
