from pathlib import Path
import html as html_lib
import json

import streamlit as st
import streamlit.components.v1 as components

from src.answer_guard import refusal_message, should_refuse
from src.bandit_recommender import recommend_concept, update_feedback
from src.chunker import chunk_pages
from src.config import get_mode
from src.document_loader import load_pdf
from src.generator import answer_question, summarize_document
from src.graph_store import expand_with_graph, explain_graph_expansion
from src.miniranker import rerank
from src.retriever import retrieve
from src.schemas import DocumentPage
from src.study_tools import generate_quiz
from src.vector_store import build_index, load_vector_store, vector_store_exists


st.set_page_config(page_title="CourseMind", layout="wide")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def default_pages() -> list[DocumentPage]:
    return [
        DocumentPage(
            file_name="内置中文课程示例",
            page=1,
            text=(
                "CourseMind 是一个面向中文课程资料的智能学习助手。"
                "系统支持 PDF 解析、中文 Chunk 切分、RAG 检索、原文引用、"
                "文档总结、自动出题和复习推荐。检索阶段结合向量检索、"
                "BM25 关键词检索和 GraphRAG-lite 图扩展。MiniRanker 会综合 "
                "dense_score、bm25_score、graph_score、同页奖励、同章节奖励和文本特征，"
                "对候选片段重新排序。Bandit 模块根据学生答题反馈推荐薄弱知识点。"
            ),
        )
    ]


@st.cache_resource(show_spinner=False)
def build_knowledge_base(file_path: str | None):
    if file_path:
        pages = load_pdf(file_path)
        chunks = chunk_pages(pages)
        index = build_index(chunks)
        return pages, chunks, index, "上传文件临时索引"

    if vector_store_exists():
        chunks, index = load_vector_store()
        pages = pages_from_chunks(chunks)
        return pages, chunks, index, "data/indexes/faiss.index"

    pages = default_pages()
    chunks = chunk_pages(pages)
    index = build_index(chunks)
    return pages, chunks, index, "内置示例"


def pages_from_chunks(chunks):
    page_text: dict[tuple[str, int], list[str]] = {}
    for chunk in chunks:
        page_text.setdefault((chunk.file_name, chunk.page), []).append(chunk.text)
    return [
        DocumentPage(file_name=file_name, page=page, text="\n".join(parts))
        for (file_name, page), parts in sorted(page_text.items())
    ]


def graph_expansion_rows(expanded, seeds, chunks, seed_ids):
    rows = []
    for item in expanded:
        if item.chunk.chunk_id in seed_ids:
            continue
        relations = explain_graph_expansion(item.chunk, seeds, chunks, hops=1)
        for relation in relations:
            rows.append(
                {
                    "扩展chunk": relation.candidate_chunk_id,
                    "种子chunk": relation.seed_chunk_id,
                    "关系": relation.relation,
                    "分数": round(relation.score, 3),
                    "原因": relation.reason,
                    "文件": item.chunk.file_name,
                    "页码": item.chunk.page,
                }
            )
    return rows


def graph_reason(chunk_id, rows):
    reasons = [row["原因"] for row in rows if row["扩展chunk"] == chunk_id]
    return "；".join(reasons[:3])


def graph_visualization_html(query, seeds, rows, ranked, chunks):
    try:
        return react_force_graph_visualization_html(query, seeds, rows, ranked, chunks)
    except Exception:
        return static_graph_visualization_html(seeds, rows)


def react_force_graph_visualization_html(
    query,
    seeds,
    rows,
    ranked,
    chunks,
    max_seed_nodes=5,
    max_expanded_nodes=8,
    max_edges=14,
):
    if not rows:
        return "", 0

    seed_ids = [item.chunk.chunk_id for item in seeds[:max_seed_nodes]]
    seed_id_set = set(seed_ids)
    ranked_ids = {item.chunk.chunk_id for item in ranked}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    sorted_rows = sorted(rows, key=lambda row: row["分数"], reverse=True)

    selected_rows = []
    expanded_ids = []
    for row in sorted_rows:
        if row["种子chunk"] not in seed_id_set:
            continue
        expanded_id = row["扩展chunk"]
        if expanded_id not in expanded_ids:
            if len(expanded_ids) >= max_expanded_nodes:
                continue
            expanded_ids.append(expanded_id)
        selected_rows.append(row)
        if len(selected_rows) >= max_edges:
            break

    if not selected_rows:
        return "", 0

    nodes_by_id = {}
    links = []

    def add_node(node_id, node_type, label, tooltip="", **extra):
        existing = nodes_by_id.get(node_id)
        style = node_style(node_type)
        payload = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "tooltip": tooltip or label,
            "color": style["color"],
            "val": style["val"],
            "shape": style["shape"],
        }
        payload.update(extra)
        if existing:
            if existing["type"] != "evidence" or node_type == "evidence":
                existing.update(payload)
            return
        nodes_by_id[node_id] = payload

    def add_link(source, target, relation, score=1.0, reason=""):
        if source not in nodes_by_id or target not in nodes_by_id:
            return
        links.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "label": edge_label(relation),
                "color": edge_color(relation),
                "width": 1.3 + min(float(score), 1.0) * 2.4,
                "score": round(float(score), 3),
                "reason": reason,
            }
        )

    query_id = "query:current"
    add_node(
        query_id,
        "query",
        label="Query",
        tooltip=f"Query\n{query}",
    )

    related_chunk_ids = set(seed_ids) | set(expanded_ids) | ranked_ids
    for chunk_id in related_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        node_type = "evidence" if chunk_id in ranked_ids else "chunk"
        add_node(
            chunk_id,
            node_type,
            label=short_label(chunk_id, 16),
            tooltip=chunk_hover_title(chunk, plain_text=True),
            file_name=chunk.file_name,
            page=chunk.page,
            concepts=chunk.concepts[:8],
            text=" ".join(chunk.text.split())[:260],
        )
        file_id = f"doc:{chunk.file_name}"
        page_id = f"page:{chunk.file_name}:{chunk.page}"
        add_node(
            file_id,
            "document",
            label=short_label(chunk.file_name, 18),
            tooltip=f"Document\n{chunk.file_name}",
        )
        add_node(
            page_id,
            "page",
            label=f"Page {chunk.page}",
            tooltip=f"Page\n{chunk.file_name}\npage: {chunk.page}",
        )
        add_link(file_id, page_id, "contains", 0.35)
        add_link(page_id, chunk_id, "contains", 0.45)
        for concept in chunk.concepts[:2]:
            concept_id = f"concept:{concept}"
            add_node(
                concept_id,
                "concept",
                label=short_label(concept, 14),
                tooltip=f"Concept\n{concept}",
            )
            add_link(chunk_id, concept_id, "concept", 0.55)

    for seed_id in seed_ids:
        if seed_id in nodes_by_id:
            if seed_id not in ranked_ids:
                nodes_by_id[seed_id].update(node_style("seed"))
                nodes_by_id[seed_id]["type"] = "seed"
            add_link(query_id, seed_id, "seed", 0.9)

    for row in selected_rows:
        seed_id = row["种子chunk"]
        expanded_id = row["扩展chunk"]
        if seed_id in nodes_by_id and expanded_id in nodes_by_id:
            add_link(
                seed_id,
                expanded_id,
                row["关系"],
                float(row["分数"]),
                row["原因"],
            )

    graph_data = json.dumps(
        {"nodes": list(nodes_by_id.values()), "links": links},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = react_force_graph_html(graph_data)
    return html, 760


def node_style(node_type):
    if node_type == "query":
        return {"color": "#94a3b8", "shape": "diamond", "val": 8}
    if node_type == "document":
        return {"color": "#60a5fa", "shape": "square", "val": 6}
    if node_type == "page":
        return {"color": "#34d399", "shape": "square", "val": 5}
    if node_type == "concept":
        return {"color": "#c084fc", "shape": "circle", "val": 5}
    if node_type == "seed":
        return {"color": "#f59e0b", "shape": "circle", "val": 7}
    if node_type == "evidence":
        return {"color": "#ef4444", "shape": "star", "val": 10}
    return {"color": "#f59e0b", "shape": "circle", "val": 4}


def edge_color(relation):
    return {
        "seed": "#94a3b8",
        "contains": "#475569",
        "concept": "#c084fc",
        "same_page": "#60a5fa",
        "same_chapter": "#a78bfa",
        "adjacent": "#22d3ee",
        "shared_concept": "#34d399",
    }.get(relation, "#94a3b8")


def edge_label(relation):
    return relation if relation in {"seed", "same_page", "adjacent"} else ""


def chunk_hover_title(chunk, plain_text=False):
    concepts = ", ".join(chunk.concepts[:8])
    text = " ".join(chunk.text.split())[:260]
    if plain_text:
        return (
            f"chunk_id: {chunk.chunk_id}\n"
            f"file: {chunk.file_name}\n"
            f"page: {chunk.page}\n"
            f"concepts: {concepts}\n\n"
            f"text: {text}..."
        )
    return (
        f"<b>chunk_id:</b> {html_lib.escape(chunk.chunk_id)}<br>"
        f"<b>file:</b> {html_lib.escape(chunk.file_name)}<br>"
        f"<b>page:</b> {chunk.page}<br>"
        f"<b>concepts:</b> {html_lib.escape(concepts)}<br><br>"
        f"<b>text:</b> {html_lib.escape(text)}..."
    )


def react_force_graph_html(graph_data):
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #090d18;
      color: #e5e7eb;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      height: 748px;
      display: grid;
      grid-template-rows: auto 1fr;
      background:
        radial-gradient(circle at 12% 10%, rgba(96, 165, 250, 0.18), transparent 26%),
        radial-gradient(circle at 86% 26%, rgba(192, 132, 252, 0.14), transparent 28%),
        #090d18;
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 8px;
      box-sizing: border-box;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px 10px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    }}
    .title {{
      font-size: 16px;
      font-weight: 700;
      color: #f8fafc;
    }}
    .hint {{
      margin-top: 4px;
      font-size: 12px;
      color: #94a3b8;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px 12px;
      max-width: 520px;
      font-size: 12px;
      color: #cbd5e1;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 12px currentColor;
    }}
    .main {{
      position: relative;
      min-height: 0;
    }}
    #graph {{
      position: absolute;
      inset: 0;
    }}
    .inspector {{
      position: absolute;
      right: 14px;
      top: 14px;
      width: min(340px, calc(100% - 28px));
      max-height: 276px;
      overflow: auto;
      padding: 12px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.88);
      box-shadow: 0 18px 46px rgba(0, 0, 0, 0.34);
      backdrop-filter: blur(10px);
      box-sizing: border-box;
      font-size: 12px;
      line-height: 1.55;
      color: #dbeafe;
    }}
    .inspector-title {{
      margin-bottom: 6px;
      font-size: 13px;
      font-weight: 700;
      color: #f8fafc;
      word-break: break-all;
    }}
    .inspector-meta {{
      color: #bfdbfe;
      word-break: break-word;
    }}
    .inspector-text {{
      margin-top: 8px;
      color: #cbd5e1;
      word-break: break-word;
    }}
    .loading {{
      padding: 24px;
      color: #cbd5e1;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="title">GraphRAG 当前问题子图</div>
        <div class="hint">React Force Graph 渲染；拖动节点、滚轮缩放，点击 chunk 查看证据信息。</div>
      </div>
      <div class="legend">
        <span class="legend-item"><span class="dot" style="color:#94a3b8;background:#94a3b8"></span>Query</span>
        <span class="legend-item"><span class="dot" style="color:#60a5fa;background:#60a5fa"></span>Document</span>
        <span class="legend-item"><span class="dot" style="color:#34d399;background:#34d399"></span>Page</span>
        <span class="legend-item"><span class="dot" style="color:#f59e0b;background:#f59e0b"></span>Chunk</span>
        <span class="legend-item"><span class="dot" style="color:#c084fc;background:#c084fc"></span>Concept</span>
        <span class="legend-item"><span class="dot" style="color:#ef4444;background:#ef4444"></span>Final Evidence</span>
      </div>
    </div>
    <div class="main">
      <div id="graph"><div class="loading">正在加载 React Force Graph...</div></div>
      <div id="inspector" class="inspector">
        <div class="inspector-title">点击一个节点查看详情</div>
        <div class="inspector-meta">红色节点是 MiniRanker 最终采用的证据；橙色节点是原始 seed chunk；绿色和紫色节点展示页码与知识点连接。</div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/react-force-graph-2d"></script>
  <script>
    const graphData = {graph_data};
    const nodeStyles = {{
      query: {{ labelColor: "#f8fafc", stroke: "#cbd5e1" }},
      document: {{ labelColor: "#bfdbfe", stroke: "#93c5fd" }},
      page: {{ labelColor: "#bbf7d0", stroke: "#86efac" }},
      concept: {{ labelColor: "#e9d5ff", stroke: "#d8b4fe" }},
      seed: {{ labelColor: "#fde68a", stroke: "#fbbf24" }},
      evidence: {{ labelColor: "#fecaca", stroke: "#fca5a5" }},
      chunk: {{ labelColor: "#fde68a", stroke: "#fbbf24" }}
    }};

    function drawStar(ctx, x, y, radius, color) {{
      ctx.beginPath();
      for (let i = 0; i < 10; i += 1) {{
        const angle = Math.PI / 5 * i - Math.PI / 2;
        const r = i % 2 === 0 ? radius : radius * 0.48;
        const px = x + Math.cos(angle) * r;
        const py = y + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }}
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    }}

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[ch]));
    }}

    function renderInspector(node) {{
      const panel = document.getElementById("inspector");
      if (!node) return;
      const concepts = Array.isArray(node.concepts) && node.concepts.length
        ? `<div class="inspector-meta">concepts: ${{escapeHtml(node.concepts.join(", "))}}</div>`
        : "";
      const file = node.file_name ? `<div class="inspector-meta">file: ${{escapeHtml(node.file_name)}}</div>` : "";
      const page = node.page ? `<div class="inspector-meta">page: ${{escapeHtml(node.page)}}</div>` : "";
      const text = node.text ? `<div class="inspector-text">${{escapeHtml(node.text)}}...</div>` : "";
      panel.innerHTML = `
        <div class="inspector-title">${{escapeHtml(node.label || node.id)}}</div>
        <div class="inspector-meta">type: ${{escapeHtml(node.type)}}</div>
        ${{file}}${{page}}${{concepts}}${{text}}
      `;
    }}

    function drawNode(node, ctx, globalScale) {{
      const radius = Math.max(4.5, Math.sqrt(node.val || 4) * 2.2);
      const color = node.color || "#f59e0b";
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = node.type === "evidence" ? 18 : 8;

      if (node.shape === "square") {{
        ctx.fillStyle = color;
        ctx.beginPath();
        if (ctx.roundRect) {{
          ctx.roundRect(node.x - radius, node.y - radius, radius * 2, radius * 2, 3);
        }} else {{
          ctx.rect(node.x - radius, node.y - radius, radius * 2, radius * 2);
        }}
        ctx.fill();
      }} else if (node.shape === "diamond") {{
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(node.x, node.y - radius * 1.25);
        ctx.lineTo(node.x + radius * 1.25, node.y);
        ctx.lineTo(node.x, node.y + radius * 1.25);
        ctx.lineTo(node.x - radius * 1.25, node.y);
        ctx.closePath();
        ctx.fill();
      }} else if (node.shape === "star") {{
        drawStar(ctx, node.x, node.y, radius * 1.45, color);
      }} else {{
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
        ctx.fill();
      }}

      ctx.shadowBlur = 0;
      ctx.lineWidth = node.type === "evidence" ? 2.2 : 1.2;
      ctx.strokeStyle = nodeStyles[node.type]?.stroke || "#fbbf24";
      ctx.stroke();

      const fontSize = Math.max(9, 13 / globalScale);
      const alwaysLabel = ["query", "evidence", "seed", "page"].includes(node.type);
      const detailLabel = globalScale > 1.55 && ["document", "chunk"].includes(node.type);
      const conceptLabel = globalScale > 2.2 && node.type === "concept";
      if (alwaysLabel || detailLabel || conceptLabel) {{
        const label = node.label || node.id;
        ctx.font = `${{fontSize}}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
        const labelWidth = ctx.measureText(label).width + 10;
        ctx.fillStyle = "rgba(9, 13, 24, 0.62)";
        ctx.fillRect(node.x - labelWidth / 2, node.y + radius + 2, labelWidth, fontSize + 6);
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = nodeStyles[node.type]?.labelColor || "#e5e7eb";
        ctx.fillText(label, node.x, node.y + radius + 5);
      }}
      ctx.restore();
    }}

    function drawLinkLabel(link, ctx, globalScale) {{
      if (!link.label || globalScale > 1.9) return;
      const start = link.source;
      const end = link.target;
      if (!start || !end || start.x === undefined || end.x === undefined) return;
      const textPos = {{ x: start.x + (end.x - start.x) * 0.5, y: start.y + (end.y - start.y) * 0.5 }};
      ctx.save();
      ctx.font = `${{Math.max(8, 10 / globalScale)}}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(226, 232, 240, 0.72)";
      ctx.fillText(link.label, textPos.x, textPos.y);
      ctx.restore();
    }}

    let fgRef = null;
    let forceConfigured = false;
    let fitDone = false;

    function boot() {{
      const ForceGraph2D = window.ForceGraph2D || window.ReactForceGraph2D || window["react-force-graph-2d"];
      const rootEl = document.getElementById("graph");
      if (!window.React || !window.ReactDOM || !ForceGraph2D) {{
        rootEl.innerHTML = '<div class="loading">React Force Graph 加载失败。请检查网络或 CDN 访问权限。</div>';
        return;
      }}
      const Graph = React.createElement(ForceGraph2D, {{
        ref: graph => {{
          fgRef = graph;
          if (graph && !forceConfigured) {{
            forceConfigured = true;
            setTimeout(() => {{
              try {{
                graph.d3Force("charge").strength(-190);
                graph.d3Force("link").distance(link => {{
                  if (link.relation === "contains") return 62;
                  if (link.relation === "concept") return 52;
                  if (link.relation === "seed") return 118;
                  return 104;
                }});
                graph.d3ReheatSimulation();
              }} catch (err) {{}}
            }}, 0);
          }}
        }},
        graphData,
        backgroundColor: "rgba(0,0,0,0)",
        nodeId: "id",
        nodeLabel: node => node.tooltip || node.label || node.id,
        nodeVal: node => node.val || 4,
        linkLabel: link => [link.relation, link.score ? `score: ${{link.score}}` : "", link.reason || ""].filter(Boolean).join("\\n"),
        linkColor: link => link.color || "#64748b",
        linkWidth: link => link.width || 1.5,
        linkDirectionalArrowLength: 4,
        linkDirectionalArrowRelPos: 1,
        linkDirectionalParticles: link => link.relation === "seed" ? 2 : 0,
        linkDirectionalParticleWidth: link => link.relation === "seed" ? 2.4 : 1.6,
        cooldownTicks: 90,
        d3AlphaDecay: 0.035,
        d3VelocityDecay: 0.28,
        onNodeClick: renderInspector,
        onNodeHover: node => {{ document.body.style.cursor = node ? "pointer" : "default"; }},
        onEngineStop: () => {{
          if (fgRef && !fitDone) {{
            fitDone = true;
            setTimeout(() => fgRef.zoomToFit(360, 72), 50);
          }}
        }},
        nodeCanvasObject: drawNode,
        linkCanvasObjectMode: () => "after",
        linkCanvasObject: drawLinkLabel
      }});
      if (ReactDOM.createRoot) {{
        ReactDOM.createRoot(rootEl).render(Graph);
      }} else {{
        ReactDOM.render(Graph, rootEl);
      }}
    }}
    window.addEventListener("load", boot);
  </script>
</body>
</html>
"""


def static_graph_visualization_html(seeds, rows, max_seed_nodes=5, max_expanded_nodes=8, max_edges=14):
    if not rows:
        return "", 0

    seed_ids = [item.chunk.chunk_id for item in seeds[:max_seed_nodes]]
    seed_id_set = set(seed_ids)
    sorted_rows = sorted(rows, key=lambda row: row["分数"], reverse=True)

    expanded_ids = []
    edges = []
    for row in sorted_rows:
        if row["种子chunk"] not in seed_id_set:
            continue
        expanded_id = row["扩展chunk"]
        if expanded_id not in expanded_ids:
            if len(expanded_ids) >= max_expanded_nodes:
                continue
            expanded_ids.append(expanded_id)
        edges.append(row)
        if len(edges) >= max_edges:
            break

    if not edges:
        return "", 0

    seed_y = {chunk_id: 86 + index * 92 for index, chunk_id in enumerate(seed_ids)}
    expanded_y = {
        chunk_id: 86 + index * 92 for index, chunk_id in enumerate(expanded_ids)
    }
    height = max(360, 130 + 92 * max(len(seed_ids), len(expanded_ids)))
    width = 1120
    seed_x = 170
    expanded_x = 840
    line_start_x = 320
    line_end_x = 690

    relation_color = {
        "same_page": "#2563eb",
        "same_chapter": "#7c3aed",
        "adjacent": "#0891b2",
        "shared_concept": "#16a34a",
    }

    seed_cards = []
    seed_lookup = {item.chunk.chunk_id: item.chunk for item in seeds}
    for chunk_id in seed_ids:
        chunk = seed_lookup[chunk_id]
        seed_cards.append(
            svg_node(
                seed_x,
                seed_y[chunk_id],
                "种子",
                short_label(chunk_id),
                f"{chunk.file_name} p.{chunk.page}",
                "#e0f2fe",
                "#0369a1",
            )
        )

    expanded_cards = []
    expanded_meta = {row["扩展chunk"]: row for row in rows}
    for chunk_id in expanded_ids:
        row = expanded_meta[chunk_id]
        expanded_cards.append(
            svg_node(
                expanded_x,
                expanded_y[chunk_id],
                "扩展",
                short_label(chunk_id),
                f"{row['文件']} p.{row['页码']}",
                "#dcfce7",
                "#15803d",
            )
        )

    edge_lines = []
    for row in edges:
        source_y = seed_y[row["种子chunk"]]
        target_y = expanded_y[row["扩展chunk"]]
        color = relation_color.get(row["关系"], "#64748b")
        mid_x = (line_start_x + line_end_x) / 2
        mid_y = (source_y + target_y) / 2
        edge_lines.append(
            f"""
            <path d="M {line_start_x} {source_y} C 460 {source_y}, 550 {target_y}, {line_end_x} {target_y}"
                  fill="none" stroke="{color}" stroke-width="2.4" opacity="0.72" marker-end="url(#arrow)" />
            <rect x="{mid_x - 58}" y="{mid_y - 15}" width="116" height="30" rx="6"
                  fill="white" stroke="{color}" stroke-width="1" opacity="0.96" />
            <text x="{mid_x}" y="{mid_y + 5}" text-anchor="middle" class="edge-label" fill="{color}">
              {html_lib.escape(row["关系"])}
            </text>
            """
        )

    concept_chips = concept_chip_html(edges)
    svg = f"""
    <div class="graph-wrap">
      <div class="graph-title">GraphRAG 节点关系可视化</div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
                  orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
          </marker>
        </defs>
        <text x="{seed_x}" y="34" text-anchor="middle" class="column-title">原始检索种子 chunk</text>
        <text x="{expanded_x}" y="34" text-anchor="middle" class="column-title">GraphRAG 扩展 chunk</text>
        {''.join(edge_lines)}
        {''.join(seed_cards)}
        {''.join(expanded_cards)}
      </svg>
      {concept_chips}
      <div class="legend">
        <span><b style="color:#2563eb">same_page</b> 同页</span>
        <span><b style="color:#7c3aed">same_chapter</b> 同章节</span>
        <span><b style="color:#0891b2">adjacent</b> 相邻片段</span>
        <span><b style="color:#16a34a">shared_concept</b> 共享知识点</span>
      </div>
    </div>
    <style>
      .graph-wrap {{
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        background: #f8fafc;
        padding: 12px 14px 10px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .graph-title {{
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 6px;
      }}
      .column-title {{
        font-size: 14px;
        font-weight: 700;
        fill: #334155;
      }}
      .node-title {{
        font-size: 12px;
        font-weight: 700;
      }}
      .node-label {{
        font-size: 11px;
        fill: #0f172a;
      }}
      .node-meta {{
        font-size: 10px;
        fill: #475569;
      }}
      .edge-label {{
        font-size: 11px;
        font-weight: 700;
      }}
      .concepts {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 2px 4px;
      }}
      .concept-chip {{
        border: 1px solid #bbf7d0;
        background: #f0fdf4;
        color: #166534;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
      }}
      .legend {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        color: #475569;
        font-size: 12px;
        margin-top: 8px;
      }}
    </style>
    """
    return svg, height + 130


def svg_node(x, y, badge, title, meta, fill, stroke):
    return f"""
    <g>
      <rect x="{x - 135}" y="{y - 34}" width="270" height="68" rx="8"
            fill="{fill}" stroke="{stroke}" stroke-width="1.6" />
      <text x="{x - 118}" y="{y - 13}" class="node-title" fill="{stroke}">
        {html_lib.escape(badge)}
      </text>
      <text x="{x - 118}" y="{y + 7}" class="node-label">
        {html_lib.escape(title)}
      </text>
      <text x="{x - 118}" y="{y + 25}" class="node-meta">
        {html_lib.escape(short_label(meta, 30))}
      </text>
    </g>
    """


def concept_chip_html(edges):
    concepts = []
    for row in edges:
        reason = row["原因"]
        marker = "共享知识点："
        if marker not in reason:
            continue
        for concept in reason.split(marker, 1)[1].split("、"):
            concept = concept.strip()
            if concept and concept not in concepts:
                concepts.append(concept)
    if not concepts:
        return ""
    chips = "".join(
        f'<span class="concept-chip">concept: {html_lib.escape(short_label(concept, 18))}</span>'
        for concept in concepts[:10]
    )
    return f'<div class="concepts">{chips}</div>'


def short_label(value, max_length=24):
    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


st.sidebar.title("CourseMind")
st.sidebar.caption(f"运行模式：{get_mode()}")
uploaded = st.sidebar.file_uploader("上传课程 PDF 或文本", type=["pdf", "txt", "md"])

file_path = None
if uploaded:
    target = RAW_DIR / uploaded.name
    target.write_bytes(uploaded.getbuffer())
    file_path = str(target)

pages, chunks, vector_index, data_source = build_knowledge_base(file_path)
st.sidebar.metric("页数", len(pages))
st.sidebar.metric("Chunks", len(chunks))
st.sidebar.caption(f"数据来源：{data_source}")

tab_qa, tab_summary, tab_quiz, tab_review, tab_status = st.tabs(
    ["问答", "总结", "出题", "复习", "状态"]
)

with tab_qa:
    query = st.text_input("问题", "神经网络中的激活函数有什么作用？")
    top_k = st.slider("Top-K 证据", 1, 10, 5)
    if st.button("提问", type="primary"):
        retrieved = retrieve(query, chunks, top_k=top_k, index=vector_index)
        expanded = expand_with_graph(retrieved, chunks, hops=1)
        ranked = rerank(query, expanded, top_k=top_k)
        seed_ids = {item.chunk.chunk_id for item in retrieved}
        graph_explanations = graph_expansion_rows(expanded, retrieved, chunks, seed_ids)

        if should_refuse(query, ranked):
            st.warning(refusal_message())
        else:
            try:
                result = answer_question(query, ranked)
            except Exception as exc:
                st.error(f"回答生成失败：{exc}")
                st.caption(
                    "如果使用 DeepSeek，请在 .env 中设置 LLM_PROVIDER=deepseek "
                    "并填写 DEEPSEEK_API_KEY，然后重启 Streamlit。"
                )
            else:
                st.subheader("回答")
                st.write(result["answer"])
                st.subheader("引用来源")
                st.dataframe(result["citations"], use_container_width=True)

        st.subheader("检索与重排序结果")
        st.dataframe(
            [
                {
                    "rank": i + 1,
                    "来源": "原始检索" if item.chunk.chunk_id in seed_ids else "GraphRAG扩展",
                    "chunk_id": item.chunk.chunk_id,
                    "文件": item.chunk.file_name,
                    "页码": item.chunk.page,
                    "dense": round(item.dense_score, 3),
                    "bm25": round(item.bm25_score, 3),
                    "graph": round(item.graph_score, 3),
                    "ranker": round(item.ranker_score, 3),
                    "GraphRAG原因": graph_reason(item.chunk.chunk_id, graph_explanations),
                    "文本": item.chunk.text[:120],
                }
                for i, item in enumerate(ranked)
            ],
            use_container_width=True,
        )

        if graph_explanations:
            graph_html, graph_height = graph_visualization_html(
                query, retrieved, graph_explanations, ranked, chunks
            )
            if graph_html:
                st.subheader("GraphRAG 节点关系图")
                components.html(graph_html, height=graph_height, scrolling=True)
            st.subheader("GraphRAG 扩展解释")
            st.dataframe(graph_explanations, use_container_width=True)

with tab_summary:
    if st.button("生成总结"):
        st.write(summarize_document(chunks))

with tab_quiz:
    num_questions = st.slider("题目数量", 1, 5, 3)
    quiz_items = generate_quiz(chunks, num_questions=num_questions)
    for idx, item in enumerate(quiz_items, start=1):
        st.markdown(f"**Q{idx}. {item.question}**")
        choice = st.radio("请选择", item.options, key=f"quiz_{idx}")
        correct = choice == item.answer
        col1, col2 = st.columns(2)
        with col1:
            if st.button("提交", key=f"submit_{idx}"):
                update_feedback(item.concept, correct)
                st.success("回答正确" if correct else f"回答错误。正确答案：{item.answer}")
        with col2:
            st.caption(f"知识点：{item.concept} | 来源：{item.source_chunk_id}")
        st.write(item.explanation)

with tab_review:
    rec = recommend_concept()
    st.metric("推荐复习知识点", rec["concept"])
    st.write(rec["reason"])
    st.json(rec)

with tab_status:
    st.subheader("系统状态")
    st.write(
        {
            "mode": get_mode(),
            "pages": len(pages),
            "chunks": len(chunks),
            "vector_index": data_source,
            "fallback_ready": True,
        }
    )
