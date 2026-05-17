from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = "data/agent/kg_viewer_data.json"
DEFAULT_OUTPUT = "outputs/agent/kg_viewer.html"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"KG viewer data file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(data), encoding="utf-8")
    print(f"[kg-viewer] input={input_path}")
    print(f"[kg-viewer] output={output_path}")
    print(f"[kg-viewer] nodes={len(data.get('nodes') or [])}")
    print(f"[kg-viewer] edges={len(data.get('edges') or [])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a single-file HTML KG viewer.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_html(data: dict) -> str:
    graph_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Aquaculture Literature Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #172033;
      --muted: #667085;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    header {{
      padding: 14px 18px;
      background: #102033;
      color: white;
      border-bottom: 1px solid #0b1624;
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 21px;
      font-weight: 700;
    }}
    header .hint {{
      font-size: 12px;
      color: #d9e6f2;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1.2fr) minmax(180px, 1fr) minmax(180px, 1fr) 180px 190px;
      gap: 10px;
      padding: 12px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      align-items: start;
    }}
    .control label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
      color: #344054;
    }}
    input[type="text"], input[type="number"] {{
      width: 100%;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 13px;
    }}
    .checklist {{
      max-height: 88px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px;
      background: #fbfcfe;
      font-size: 12px;
    }}
    .checklist label {{
      display: block;
      font-weight: 400;
      margin: 2px 0;
    }}
    .buttons {{
      display: flex;
      gap: 8px;
      padding-top: 18px;
    }}
    button {{
      padding: 8px 10px;
      border: 1px solid #0f766e;
      background: var(--accent);
      color: white;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
    }}
    button.secondary {{
      background: white;
      color: var(--accent);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 370px;
      min-height: calc(100vh - 238px);
    }}
    #network {{
      height: calc(100vh - 235px);
      min-height: 560px;
      background: white;
      border-right: 1px solid var(--line);
    }}
    #detail {{
      padding: 14px;
      background: #fbfcfe;
      overflow: auto;
      max-height: calc(100vh - 235px);
    }}
    #detail h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .kv {{
      margin: 8px 0;
      font-size: 13px;
    }}
    .kv b {{
      display: block;
      color: #344054;
      font-size: 12px;
      margin-bottom: 2px;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 6px;
      margin: 2px;
      border-radius: 999px;
      background: #eef2f7;
      font-size: 12px;
    }}
    .evidence {{
      padding: 8px;
      margin: 8px 0;
      border-left: 3px solid var(--accent);
      background: white;
      border-radius: 4px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .empty {{
      padding: 24px;
      color: #b42318;
      font-weight: 700;
    }}
    footer {{
      padding: 12px 14px;
      background: var(--panel);
      border-top: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .toolbar {{ grid-template-columns: 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      #network {{ height: 560px; border-right: 0; border-bottom: 1px solid var(--line); }}
      #detail {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Aquaculture Literature Knowledge Graph</h1>
    <div class="hint">If the graph cannot load, check whether the browser can access the vis-network CDN; this can later be changed to a local dependency. Edges are rule-extracted candidate relations and evidence pointers, not manually verified biological facts.</div>
  </header>

  <section class="toolbar">
    <div class="control">
      <label for="searchBox">Search entity</label>
      <input id="searchBox" type="text" placeholder="e.g. dna methylation, zebrafish, cyp19a" />
    </div>
    <div class="control">
      <label>Entity types</label>
      <div id="entityFilters" class="checklist"></div>
    </div>
    <div class="control">
      <label>Predicates</label>
      <div id="predicateFilters" class="checklist"></div>
    </div>
    <div class="control">
      <label for="minEvidence">Min evidence_count</label>
      <input id="minEvidence" type="number" min="1" step="1" value="1" />
    </div>
    <div class="buttons">
      <button id="resetBtn" class="secondary">Reset View</button>
      <button id="fitBtn">Fit Graph</button>
    </div>
  </section>

  <main>
    <div id="network"></div>
    <aside id="detail">
      <h2>Details</h2>
      <p>Click a node or edge to inspect provenance.</p>
    </aside>
  </main>

  <footer id="stats"></footer>

  <script>
    const graphData = {graph_json};
    const entityColors = {{
      species: "#2f80ed",
      gene: "#9b51e0",
      protein: "#7b61ff",
      phenotype: "#f2994a",
      tissue: "#27ae60",
      method: "#00a3a3",
      chemical: "#eb5757",
      environmental_factor: "#8a6d3b",
      disease: "#c0392b",
      default: "#667085"
    }};

    let network;
    let nodesDataSet;
    let edgesDataSet;
    let allNodes = [];
    let allEdges = [];

    function sizeForMentions(count) {{
      return Math.max(12, Math.min(44, 12 + Math.sqrt(count || 0) * 2.3));
    }}

    function widthForEvidence(count) {{
      return Math.max(1, Math.min(8, 1 + Math.log2((count || 1) + 1)));
    }}

    function nodeToVis(node) {{
      const color = entityColors[node.entity_type] || entityColors.default;
      return {{
        id: node.id,
        label: node.label,
        group: node.group,
        title: `${{node.label}} (${{node.entity_type}})`,
        value: node.mention_count || 1,
        size: sizeForMentions(node.mention_count),
        color: {{
          background: color,
          border: node.is_query_matched ? "#111827" : "#ffffff",
          highlight: {{ background: color, border: "#111827" }}
        }},
        borderWidth: node.is_query_matched ? 4 : 1,
        font: {{ color: "#172033", size: 13, strokeWidth: 3, strokeColor: "#ffffff" }},
        raw: node
      }};
    }}

    function edgeToVis(edge) {{
      return {{
        id: edge.id,
        from: edge.source,
        to: edge.target,
        label: edge.label,
        arrows: "to",
        width: widthForEvidence(edge.evidence_count),
        color: edge.is_retrieved ? {{ color: "#d92d20", highlight: "#b42318" }} : {{ color: "#98a2b3", highlight: "#475467" }},
        dashes: edge.is_retrieved ? false : [4, 4],
        font: {{ size: 11, align: "middle", background: "rgba(255,255,255,0.85)" }},
        smooth: {{ type: "dynamic" }},
        raw: edge
      }};
    }}

    function init() {{
      allNodes = graphData.nodes || [];
      allEdges = graphData.edges || [];
      buildFilters();
      renderStats();
      if (!allNodes.length || !allEdges.length) {{
        document.getElementById("network").innerHTML = '<div class="empty">no graph data available</div>';
        return;
      }}
      applyFilters();
      document.getElementById("searchBox").addEventListener("input", applyFilters);
      document.getElementById("minEvidence").addEventListener("input", applyFilters);
      document.getElementById("resetBtn").addEventListener("click", resetView);
      document.getElementById("fitBtn").addEventListener("click", () => network && network.fit({{ animation: true }}));
    }}

    function buildFilters() {{
      const types = [...new Set(allNodes.map(n => n.entity_type).filter(Boolean))].sort();
      const predicates = [...new Set(allEdges.map(e => e.predicate).filter(Boolean))].sort();
      fillChecklist("entityFilters", types, "entity");
      fillChecklist("predicateFilters", predicates, "predicate");
    }}

    function fillChecklist(containerId, values, prefix) {{
      const box = document.getElementById(containerId);
      box.innerHTML = values.map(value => `
        <label><input type="checkbox" data-filter="${{prefix}}" value="${{escapeHtml(value)}}" checked /> ${{escapeHtml(value)}}</label>
      `).join("");
      box.querySelectorAll("input").forEach(input => input.addEventListener("change", applyFilters));
    }}

    function checkedValues(kind) {{
      return new Set([...document.querySelectorAll(`input[data-filter="${{kind}}"]:checked`)].map(i => i.value));
    }}

    function applyFilters() {{
      const selectedTypes = checkedValues("entity");
      const selectedPredicates = checkedValues("predicate");
      const minEvidence = Number(document.getElementById("minEvidence").value || 1);
      const query = document.getElementById("searchBox").value.trim().toLowerCase();

      const visibleEdges = allEdges.filter(edge =>
        selectedPredicates.has(edge.predicate) &&
        (edge.evidence_count || 0) >= minEvidence
      );
      const edgeNodeIds = new Set();
      visibleEdges.forEach(edge => {{ edgeNodeIds.add(edge.source); edgeNodeIds.add(edge.target); }});

      let visibleNodes = allNodes.filter(node =>
        edgeNodeIds.has(node.id) &&
        selectedTypes.has(node.entity_type)
      );
      if (query) {{
        visibleNodes = visibleNodes.filter(node => (node.label || "").toLowerCase().includes(query));
        const queryNodeIds = new Set(visibleNodes.map(node => node.id));
        visibleEdges.splice(0, visibleEdges.length, ...visibleEdges.filter(edge => queryNodeIds.has(edge.source) || queryNodeIds.has(edge.target)));
        visibleEdges.forEach(edge => {{ queryNodeIds.add(edge.source); queryNodeIds.add(edge.target); }});
        visibleNodes = allNodes.filter(node => queryNodeIds.has(node.id) && selectedTypes.has(node.entity_type));
      }}

      const visibleNodeIds = new Set(visibleNodes.map(node => node.id));
      const finalEdges = visibleEdges.filter(edge => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));
      drawGraph(visibleNodes, finalEdges, query);
    }}

    function drawGraph(nodes, edges, query) {{
      nodesDataSet = new vis.DataSet(nodes.map(node => {{
        const item = nodeToVis(node);
        if (query && (node.label || "").toLowerCase().includes(query)) {{
          item.borderWidth = 5;
          item.color.border = "#f79009";
        }}
        return item;
      }}));
      edgesDataSet = new vis.DataSet(edges.map(edgeToVis));
      const container = document.getElementById("network");
      const options = {{
        physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -16000, springLength: 130 }} }},
        interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
        nodes: {{ shape: "dot" }},
        edges: {{ selectionWidth: 2 }}
      }};
      network = new vis.Network(container, {{ nodes: nodesDataSet, edges: edgesDataSet }}, options);
      network.on("click", event => {{
        if (event.nodes.length) showNode(nodesDataSet.get(event.nodes[0]).raw);
        else if (event.edges.length) showEdge(edgesDataSet.get(event.edges[0]).raw);
      }});
    }}

    function resetView() {{
      document.getElementById("searchBox").value = "";
      document.getElementById("minEvidence").value = 1;
      document.querySelectorAll(".checklist input").forEach(input => input.checked = true);
      applyFilters();
    }}

    function showNode(node) {{
      document.getElementById("detail").innerHTML = `
        <h2>${{escapeHtml(node.label)}}</h2>
        ${{kv("entity_type", node.entity_type)}}
        ${{kv("mention_count", node.mention_count)}}
        ${{kv("paper_count", node.paper_count)}}
        ${{kv("max_confidence", node.max_confidence)}}
        ${{listKv("DOIs", node.dois)}}
        ${{listKv("paper_ids", node.paper_ids)}}
        ${{listKv("passage_ids", node.passage_ids)}}
        ${{listKv("source_files", node.source_files)}}
      `;
    }}

    function showEdge(edge) {{
      const evidenceHtml = (edge.evidence || []).map(item => `
        <div class="evidence">
          <b>${{escapeHtml(item.doi || "")}}</b><br/>
          <span>${{escapeHtml(item.paper_id || "")}}</span><br/>
          <span>${{escapeHtml(item.passage_id || "")}}</span><br/>
          <p>${{escapeHtml(item.evidence_text || "")}}</p>
        </div>
      `).join("") || "<p>No evidence_text available.</p>";
      document.getElementById("detail").innerHTML = `
        <h2>${{escapeHtml(edge.subject)}} -> ${{escapeHtml(edge.predicate)}} -> ${{escapeHtml(edge.object)}}</h2>
        ${{kv("evidence_count", edge.evidence_count)}}
        ${{kv("paper_count", edge.paper_count)}}
        ${{kv("max_confidence", edge.max_confidence)}}
        ${{listKv("DOIs", edge.dois)}}
        ${{listKv("paper_ids", edge.paper_ids)}}
        ${{listKv("passage_ids", edge.passage_ids)}}
        <div class="kv"><b>evidence_text</b>${{evidenceHtml}}</div>
      `;
    }}

    function renderStats() {{
      const stats = graphData.stats || {{}};
      document.getElementById("stats").innerHTML = `
        <b>Stats</b> |
        exported_nodes=${{stats.exported_nodes || 0}},
        exported_edges=${{stats.exported_edges || 0}},
        missing_doi_edge_count=${{stats.missing_doi_edge_count || 0}},
        low_evidence_edge_count=${{stats.low_evidence_edge_count || 0}},
        isolated_node_count_in_export=${{stats.isolated_node_count_in_export || 0}}
        <br/>
        node_type_distribution=${{escapeHtml(JSON.stringify(stats.node_type_distribution || {{}}))}}
        <br/>
        predicate_distribution=${{escapeHtml(JSON.stringify(stats.predicate_distribution || {{}}))}}
      `;
    }}

    function kv(key, value) {{
      return `<div class="kv"><b>${{escapeHtml(key)}}</b>${{escapeHtml(String(value ?? ""))}}</div>`;
    }}

    function listKv(key, values) {{
      const items = (values || []).slice(0, 40).map(value => `<span class="pill">${{escapeHtml(String(value))}}</span>`).join("");
      return `<div class="kv"><b>${{escapeHtml(key)}}</b>${{items || ""}}</div>`;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    window.addEventListener("load", init);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
