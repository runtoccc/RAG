from __future__ import annotations

import traceback
from typing import Any

import streamlit as st

from query_data import query_rag, retrieve_chunks
from ui_helpers import (
    format_command_output,
    get_project_status,
    get_vector_db_status,
    run_project_script,
)


st.set_page_config(
    page_title="鱼类文献本地知识库 RAG Demo",
    page_icon="🐟",
    layout="wide",
)


def main():
    status = get_project_status()
    config = status["config"]
    retrieval_config = config["retrieval"]
    embedding_config = config["embedding"]
    llm_config = config["llm"]

    with st.sidebar:
        st.title("鱼类文献 RAG")
        st.caption("本地 PDF 知识库 + ChromaDB + DeepSeek")
        st.write(f"向量库路径：`{config['paths']['chroma_dir']}`")
        st.metric("PDF 文件数", status["pdf_count"])
        st.metric("Chroma chunks", status["vector_db"]["chunk_count"])
        st.write(f"Embedding：`{embedding_config.get('provider', 'unknown')}`")
        st.write(f"LLM：`{llm_config.get('provider', 'unknown')}`")
        st.write(f"模型：`{llm_config.get('model', 'unknown')}`")

        top_k = st.slider(
            "Top-K 检索数量",
            min_value=1,
            max_value=12,
            value=int(retrieval_config.get("top_k", 5)),
        )

        if st.button("检查 DeepSeek API", use_container_width=True):
            with st.spinner("正在检查 DeepSeek API..."):
                result = run_project_script("scripts/check_api.py", timeout=120)
            show_script_result(result, success_text="DeepSeek API 可用")

        if st.button("重建向量库", use_container_width=True):
            with st.spinner("正在重建向量库，PDF 较多时需要等待..."):
                result = run_project_script("scripts/rebuild_index.py", timeout=900)
            show_script_result(result, success_text="向量库重建完成，请刷新页面查看最新状态")

        if st.button("清空聊天历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("查看 PDF 列表"):
            if status["papers"]:
                for paper in status["papers"]:
                    st.write(f"- {paper.name}")
            else:
                st.warning("data/papers/ 下没有 PDF 文件。")

    st.title("🐟 鱼类文献本地知识库 RAG Demo")
    st.caption("传统 RAG 主流程：PDF -> chunk -> embedding -> ChromaDB -> DeepSeek -> 引用来源")

    tab_qa, tab_retrieval, tab_status, tab_future = st.tabs(
        ["文献问答", "检索片段查看", "知识库状态", "后续扩展说明"]
    )

    with tab_qa:
        render_qa_tab(top_k)

    with tab_retrieval:
        render_retrieval_tab(top_k)

    with tab_status:
        render_status_tab()

    with tab_future:
        render_future_tab()


def render_qa_tab(top_k: int):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("请输入鱼类/水产文献问题")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索文献并调用 DeepSeek 生成回答..."):
            try:
                result = query_rag(question, top_k=top_k)
            except Exception as error:
                render_error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"查询失败：{error}", "sources": []}
                )
                return

        answer = result.get("answer", "")
        sources = result.get("cited_sources", [])
        st.markdown(answer)
        render_sources(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


def render_retrieval_tab(top_k: int):
    st.subheader("只检索，不调用 LLM")
    query = st.text_input("检索 query", placeholder="例如：zebrafish DNA methylation temperature")
    if st.button("检索 top-k chunks", type="primary"):
        if not query.strip():
            st.warning("请输入检索 query。")
            return

        with st.spinner("正在检索 ChromaDB..."):
            try:
                chunks = retrieve_chunks(query, top_k=top_k)
            except Exception as error:
                render_error(error)
                return

        render_sources(chunks)


def render_status_tab():
    status = get_project_status()
    config = status["config"]
    vector_status = get_vector_db_status(config)

    col_pdf, col_chunk, col_key = st.columns(3)
    col_pdf.metric("PDF 文件数", status["pdf_count"])
    col_chunk.metric("Chroma chunks", vector_status["chunk_count"])
    col_key.metric("DeepSeek Key", "已配置" if status["deepseek_key_present"] else "未配置")

    st.subheader("路径与配置")
    st.json(
        {
            "papers_dir": str(status["papers_dir"]),
            "vector_db_path": str(vector_status["path"]),
            "vector_db_exists": vector_status["exists"],
            "chunk_size": config["chunking"]["chunk_size"],
            "chunk_overlap": config["chunking"]["chunk_overlap"],
            "top_k": config["retrieval"]["top_k"],
            "embedding_provider": config["embedding"]["provider"],
            "llm_provider": config["llm"]["provider"],
            "llm_model": config["llm"]["model"],
        }
    )

    st.subheader("PDF 文件列表")
    if status["papers"]:
        st.dataframe(
            [{"pdf_file": paper.name, "path": str(paper)} for paper in status["papers"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("data/papers/ 不存在或没有 PDF 文件。")

    st.subheader("Chroma metadata 示例")
    if vector_status["error"]:
        st.error(vector_status["error"])
    elif vector_status["metadata_examples"]:
        for index, example in enumerate(vector_status["metadata_examples"], start=1):
            with st.expander(f"metadata 示例 {index}", expanded=index == 1):
                st.json(example["metadata"])
                st.write(example["snippet"])
    else:
        st.info("暂无 metadata 示例。请先运行 python scripts/rebuild_index.py。")


def render_future_tab():
    st.subheader("当前已经实现")
    st.write("PDF -> chunk -> embedding -> ChromaDB -> RAG 问答 -> 引用来源")

    st.subheader("后续计划")
    st.write("1. 接入 scripts/incremental_update.py，实现每周新增文献入库。")
    st.write("2. 接入文献元数据爬虫，获取 title、abstract、DOI、PMID、year、journal。")
    st.write("3. 对新文献做规则筛选和 LLM/SciBERT 精筛。")
    st.write("4. 抽取实体关系，写入 kg/。")
    st.write("5. 后续再接 Neo4j，但当前不实现。")
    st.write("6. 后续再考虑 GraphRAG，但当前不实现。")


def render_sources(sources: list[dict[str, Any]]):
    if not sources:
        st.info("没有返回引用来源。")
        return

    for index, source in enumerate(sources, start=1):
        title = (
            f"来源 {index}: {source.get('pdf_file') or source.get('source_file') or 'unknown.pdf'} "
            f"p.{source.get('page_number', '未知')}"
        )
        with st.expander(title):
            st.write(f"PDF：`{source.get('pdf_file') or source.get('source_file') or 'unknown.pdf'}`")
            st.write(f"页码：`{source.get('page_number', '未知')}`")
            st.write(f"chunk_id：`{source.get('chunk_id', '未知')}`")
            st.write(f"paper_id：`{source.get('paper_id', '未知')}`")
            score = source.get("score")
            st.write(f"score：`{score:.4f}`" if isinstance(score, (int, float)) else f"score：`{score}`")
            st.text_area(
                "原文片段",
                source.get("snippet", ""),
                height=180,
                key=f"snippet_{index}_{source.get('chunk_id', '')}_{id(source)}",
            )


def show_script_result(result: dict[str, Any], success_text: str):
    output = format_command_output(result)
    if result["returncode"] == 0:
        st.success(success_text)
    else:
        st.error("命令执行失败")
    st.code(output, language="text")


def render_error(error: Exception):
    st.error(str(error))
    with st.expander("查看详细错误"):
        st.code("".join(traceback.format_exception(error)), language="text")


if __name__ == "__main__":
    main()
