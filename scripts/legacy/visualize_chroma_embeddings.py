from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from get_embedding_function import get_embedding_function
from rag_config import load_config, project_path
from ui_helpers import DEFAULT_COLLECTION_NAME


SECTION_COLORS = {
    "Abstract": "#2f80ed",
    "Introduction": "#27ae60",
    "Materials and Methods": "#8e44ad",
    "Methods": "#9b59b6",
    "Results": "#d35400",
    "Discussion": "#c0392b",
    "Conclusion": "#16a085",
    "References": "#7f8c8d",
    "Unknown": "#34495e",
}


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Generate a local HTML map of Chroma passage embeddings."
    )
    parser.add_argument("--query", default=None, help="Optional query to plot on the map.")
    parser.add_argument("--top-k", type=int, default=8, help="Query top-k chunks to connect.")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=2,
        help="Nearest-neighbor lines per chunk in the sampled map.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1200,
        help="Maximum chunks to draw. The query top-k chunks are always kept.",
    )
    parser.add_argument(
        "--output",
        default="data/vector_db/chroma_embedding_map.html",
        help="Output HTML path.",
    )
    args = parser.parse_args()

    config = load_config()
    collection = get_collection(config)
    items = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = items.get("ids") or []
    documents = items.get("documents") or []
    metadatas = [metadata or {} for metadata in (items.get("metadatas") or [])]
    raw_embeddings = items.get("embeddings")
    embeddings = np.asarray(
        raw_embeddings if raw_embeddings is not None else [], dtype=np.float32
    )

    if len(ids) == 0 or embeddings.size == 0:
        raise RuntimeError("Chroma collection is empty. Run: python scripts/rebuild_index.py")

    query_result = None
    query_embedding = None
    keep_ids: set[str] = set()
    if args.query:
        query_embedding = np.asarray(embed_query(args.query, config), dtype=np.float32)
        query_result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=args.top_k,
            include=["documents", "metadatas", "distances"],
        )
        keep_ids.update(query_result.get("ids", [[]])[0])

    selected_indexes = select_indexes(ids, metadatas, limit=args.limit, keep_ids=keep_ids)
    selected_embeddings = embeddings[selected_indexes]
    selected_ids = [ids[index] for index in selected_indexes]
    selected_documents = [documents[index] for index in selected_indexes]
    selected_metadatas = [metadatas[index] for index in selected_indexes]

    coordinates, query_coordinate = project_embeddings(selected_embeddings, query_embedding)
    points = build_points(
        selected_ids,
        selected_documents,
        selected_metadatas,
        coordinates,
    )
    edges = build_neighbor_edges(points, selected_embeddings, neighbors=args.neighbors)
    query_payload = build_query_payload(args.query, query_result, selected_ids, query_coordinate)

    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(points, edges, query_payload),
        encoding="utf-8",
    )

    summary = {
        "output": str(output_path),
        "total_chunks": len(ids),
        "drawn_chunks": len(points),
        "neighbor_edges": len(edges),
        "query": args.query,
        "query_top_k": len(query_payload.get("hits", [])) if query_payload else 0,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_collection(config: dict[str, Any]):
    chroma_path = project_path(config["paths"]["chroma_dir"])
    if not chroma_path.exists():
        raise RuntimeError(
            f"Chroma vector database does not exist: {chroma_path}. "
            "Run: python scripts/rebuild_index.py"
        )
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_collection(DEFAULT_COLLECTION_NAME)


def embed_query(query: str, config: dict[str, Any]) -> list[float]:
    return get_embedding_function(config).embed_query(query)


def select_indexes(
    ids: list[str],
    metadatas: list[dict[str, Any]],
    limit: int,
    keep_ids: set[str],
) -> list[int]:
    if len(ids) <= limit:
        return list(range(len(ids)))

    keep_indexes = [index for index, chunk_id in enumerate(ids) if chunk_id in keep_ids]
    remaining_indexes = [index for index in range(len(ids)) if index not in keep_indexes]

    section_buckets: dict[str, list[int]] = {}
    for index in remaining_indexes:
        section = str(metadatas[index].get("section") or "Unknown")
        section_buckets.setdefault(section, []).append(index)

    slots = max(0, limit - len(keep_indexes))
    sampled: list[int] = []
    random.seed(42)
    for bucket in section_buckets.values():
        random.shuffle(bucket)

    while len(sampled) < slots and section_buckets:
        for section in list(section_buckets):
            bucket = section_buckets[section]
            if not bucket:
                del section_buckets[section]
                continue
            sampled.append(bucket.pop())
            if len(sampled) >= slots:
                break

    return sorted(keep_indexes + sampled)


def project_embeddings(
    embeddings: np.ndarray, query_embedding: np.ndarray | None = None
) -> tuple[np.ndarray, list[float] | None]:
    matrix = embeddings
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    axes = components[:2].T
    raw_coordinates = centered @ axes
    min_values = raw_coordinates.min(axis=0)
    max_values = raw_coordinates.max(axis=0)
    span = np.maximum(max_values - min_values, 1e-9)
    coordinates = ((raw_coordinates - min_values) / span).astype(float)

    query_coordinate = None
    if query_embedding is not None:
        raw_query_coordinate = (query_embedding - mean) @ axes
        query_coordinate = ((raw_query_coordinate - min_values) / span).astype(float).tolist()

    return coordinates, query_coordinate


def build_points(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    coordinates: np.ndarray,
) -> list[dict[str, Any]]:
    points = []
    for index, (chunk_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
        section = str(metadata.get("section") or "Unknown")
        points.append(
            {
                "id": chunk_id,
                "x": float(coordinates[index][0]),
                "y": float(coordinates[index][1]),
                "section": section,
                "color": SECTION_COLORS.get(section, "#555555"),
                "title": metadata.get("title"),
                "page_number": metadata.get("page_number"),
                "source_file": metadata.get("source_file"),
                "is_reference_section": bool(metadata.get("is_reference_section")),
                "preview": make_preview(document),
            }
        )
    return points


def build_neighbor_edges(
    points: list[dict[str, Any]], embeddings: np.ndarray, neighbors: int
) -> list[dict[str, Any]]:
    if neighbors <= 0 or len(points) <= 1:
        return []

    normalized = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
    )
    similarity = normalized @ normalized.T
    edges = []
    seen = set()
    for index, point in enumerate(points):
        nearest = np.argsort(-similarity[index])[: neighbors + 1]
        for neighbor_index in nearest:
            if neighbor_index == index:
                continue
            left, right = sorted((index, int(neighbor_index)))
            key = (left, right)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "source": point["id"],
                    "target": points[int(neighbor_index)]["id"],
                    "similarity": float(similarity[index][neighbor_index]),
                }
            )
    return edges


def build_query_payload(
    query: str | None,
    query_result: dict[str, Any] | None,
    selected_ids: list[str],
    query_coordinate: list[float] | None,
) -> dict[str, Any] | None:
    if not query or not query_result or query_coordinate is None:
        return None

    selected = set(selected_ids)
    hits = []
    ids = query_result.get("ids", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    documents = query_result.get("documents", [[]])[0]
    distances = query_result.get("distances", [[]])[0]
    for chunk_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
        hits.append(
            {
                "id": chunk_id,
                "distance": float(distance),
                "drawn": chunk_id in selected,
                "title": (metadata or {}).get("title"),
                "section": (metadata or {}).get("section"),
                "page_number": (metadata or {}).get("page_number"),
                "source_file": (metadata or {}).get("source_file"),
                "preview": make_preview(document),
            }
        )

    return {
        "text": query,
        "x": query_coordinate[0],
        "y": query_coordinate[1],
        "hits": hits,
    }


def make_preview(text: str, max_chars: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def render_html(
    points: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    query_payload: dict[str, Any] | None,
) -> str:
    sections = sorted({point["section"] for point in points})
    payload = {
        "points": points,
        "edges": edges,
        "query": query_payload,
        "sections": sections,
        "colors": SECTION_COLORS,
    }
    # Keep JSON parseable inside a raw <script> block while preventing accidental
    # closing tags from terminating the script early.
    escaped_payload = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Chroma Embedding Map</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f7f7f2; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #d7d4c8; background: #ffffff; }}
    h1 {{ margin: 0 0 6px; font-size: 18px; }}
    .meta {{ font-size: 13px; color: #52606d; }}
    #wrap {{ display: grid; grid-template-columns: 1fr 340px; height: calc(100vh - 66px); }}
    #canvas {{ width: 100%; height: 100%; display: block; background: #fbfbf7; }}
    aside {{ overflow: auto; padding: 14px; border-left: 1px solid #d7d4c8; background: #ffffff; }}
    .legend {{ display: grid; grid-template-columns: 14px 1fr; gap: 7px 8px; align-items: center; font-size: 13px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; }}
    .hit {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e2d8; font-size: 12px; }}
    .hit strong {{ display: block; margin-bottom: 3px; }}
    #tip {{ position: fixed; pointer-events: none; display: none; max-width: 420px; padding: 10px; border: 1px solid #c9c5b8; background: #fffef9; box-shadow: 0 6px 18px rgba(0,0,0,.13); font-size: 12px; line-height: 1.35; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Chroma Embedding Map</h1>
    <div class="meta">Points are scientific passages. Gray lines connect nearest neighbors in embedding space. Red lines connect an optional query to retrieved chunks.</div>
  </header>
  <div id="wrap">
    <canvas id="canvas"></canvas>
    <aside>
      <h2 style="font-size:15px;margin:0 0 10px;">Sections</h2>
      <div id="legend" class="legend"></div>
      <div id="query"></div>
    </aside>
  </div>
  <div id="tip"></div>
  <script id="payload" type="application/json">{escaped_payload}</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const tip = document.getElementById('tip');
    const pointById = new Map(data.points.map(p => [p.id, p]));
    let screenPoints = [];

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * devicePixelRatio);
      canvas.height = Math.floor(rect.height * devicePixelRatio);
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      draw();
    }}

    function sx(x) {{ return 32 + x * (canvas.clientWidth - 64); }}
    function sy(y) {{ return 32 + (1 - y) * (canvas.clientHeight - 64); }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      screenPoints = data.points.map(p => ({{...p, sx: sx(p.x), sy: sy(p.y)}}));
      const screenById = new Map(screenPoints.map(p => [p.id, p]));

      ctx.lineWidth = 0.55;
      ctx.strokeStyle = 'rgba(93, 98, 107, 0.15)';
      for (const edge of data.edges) {{
        const a = screenById.get(edge.source);
        const b = screenById.get(edge.target);
        if (!a || !b) continue;
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
        ctx.stroke();
      }}

      if (data.query) {{
        const qx = sx(data.query.x);
        const qy = sy(data.query.y);
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = 'rgba(220, 38, 38, 0.55)';
        for (const hit of data.query.hits) {{
          const point = screenById.get(hit.id);
          if (!point) continue;
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(point.sx, point.sy);
          ctx.stroke();
        }}
        drawStar(qx, qy, 8, '#dc2626');
      }}

      for (const point of screenPoints) {{
        ctx.beginPath();
        ctx.fillStyle = point.color;
        ctx.globalAlpha = point.is_reference_section ? 0.38 : 0.82;
        ctx.arc(point.sx, point.sy, point.is_reference_section ? 2.2 : 3.1, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalAlpha = 1;
    }}

    function drawStar(x, y, r, color) {{
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {{
        const angle = -Math.PI / 2 + i * Math.PI / 5;
        const radius = i % 2 === 0 ? r : r * 0.45;
        ctx.lineTo(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius);
      }}
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }}

    function renderLegend() {{
      const legend = document.getElementById('legend');
      legend.innerHTML = data.sections.map(section => {{
        const color = data.colors[section] || '#555555';
        return `<span class="swatch" style="background:${{color}}"></span><span>${{section}}</span>`;
      }}).join('');

      const query = document.getElementById('query');
      if (!data.query) {{
        query.innerHTML = '<div class="hit">Run with <code>--query "..."</code> to draw query-to-chunk lines.</div>';
        return;
      }}
      query.innerHTML = `<h2 style="font-size:15px;margin:18px 0 8px;">Query Hits</h2>
        <div style="font-size:12px;margin-bottom:8px;">${{escapeHtml(data.query.text)}}</div>` +
        data.query.hits.map((hit, i) => `<div class="hit">
          <strong>S${{i + 1}} distance=${{hit.distance.toFixed(4)}} ${{hit.drawn ? '' : '(not drawn)'}}</strong>
          <div>${{escapeHtml(hit.title || '')}} p.${{hit.page_number || ''}} / ${{escapeHtml(hit.section || '')}}</div>
          <code>${{escapeHtml(hit.id)}}</code>
          <div>${{escapeHtml(hit.preview || '')}}</div>
        </div>`).join('');
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}

    canvas.addEventListener('mousemove', event => {{
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let nearest = null;
      let best = 999;
      for (const point of screenPoints) {{
        const d = Math.hypot(point.sx - x, point.sy - y);
        if (d < best) {{ best = d; nearest = point; }}
      }}
      if (!nearest || best > 9) {{ tip.style.display = 'none'; return; }}
      tip.style.display = 'block';
      tip.style.left = `${{event.clientX + 14}}px`;
      tip.style.top = `${{event.clientY + 14}}px`;
      tip.innerHTML = `<strong>${{escapeHtml(nearest.title || '')}}</strong><br>
        section=${{escapeHtml(nearest.section)}} page=${{nearest.page_number || ''}} ref=${{nearest.is_reference_section}}<br>
        <code>${{escapeHtml(nearest.id)}}</code><br>${{escapeHtml(nearest.preview)}}`;
    }});
    canvas.addEventListener('mouseleave', () => tip.style.display = 'none');
    addEventListener('resize', resize);
    renderLegend();
    resize();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
