# 鱼类/水产文献智能体：PlantScience.ai 对标计划

目标：在现有 `PDF -> chunk -> embedding` 地基上，逐步升级为鱼类/水产文献智能体。

原则：先保住干净可追溯的 RAG 地基，再加实体、关系、知识图谱、混合检索和证据型回答。

## 0. 环境规则

所有新增包必须装进 `rag` 环境。

```powershell
conda activate rag
cd F:\desktop\project\03_AI-ready\rag-tutorial-v2-main
python -m pip install -r requirements.txt
```

第一阶段轻量依赖：

```text
networkx
pandas
numpy
pydantic
typer
tqdm
rapidfuzz
jsonlines
```

暂不引入：

```text
Neo4j
GraphRAG 框架
reranker
LLM cleaning
自动联网下载论文
```

## 1. 当前 PDF -> chunk 地基评估

当前链路：

```text
PDF
-> GROBID TEI
-> local S2ORC-compatible object
-> metadata enrichment
-> peS2o-style clean/filter
-> OpenScholar-style 256-word passages
-> Chroma embedding
```

当前报告数字：

```text
PDF / structured records: 20
body_text 存在: 20
bib_entries 存在: 20
cite_spans: 2305
cite_spans linked to bib_entries: 2214
ref_entries: 174
ref_spans: 735
index_ready papers: 16
failed papers: 4
passages: 390
unique indexed papers: 16
avg block words: 250.51
blocks == 256 words: 374
blocks > 256 words: 0
label leak: 0
failed record leak: 0
embedding_text format errors: 0
```

结论：

```text
PDF -> chunk 地基本身是 OK 的。
chunk 规则干净，适合继续作为 embedding/RAG 地基。
主要短板不在 chunk，而在 metadata、strict filtering、OCR fallback、KG/Agent 层。
```

## 2. 和 PlantScience.ai / AutoSKG 的对比

PlantScience.ai 公开方法要点：

```text
PDF/OCR -> raw text corpus
unique hash ID
encoding/character/linebreak cleanup
lowercase for LLM efficiency
1500-token windows + 200-token overlap
LLM NER
LLM relationship extraction
entity/relationship merge
domain KG
graph-based RAG
citation-grounded answer
continuous update
```

## 3. 当前比 PlantScience.ai 更好的地方

这里的“更好”只限 PDF -> chunk / embedding 地基，不代表整体智能体超过 PlantScience.ai。

| 项目 | 当前项目优势 |
| --- | --- |
| PDF 解析 | 用 GROBID TEI，保留 title、abstract、body_text、sections、bib_entries、cite_spans、ref_spans |
| retrieval chunk | 已对齐 OpenScholar-style：main_text、whitespace split、title prefix、无 overlap、无 section label |
| chunk 洁净度 | 390 个 passages 中 block >256 为 0，标签泄漏为 0，failed record 泄漏为 0 |
| 入库门禁 | 只让 index_ready records 进入 passage/Chroma，不为了数量牺牲质量 |
| 可追溯性 | passage_id、paper_id、source_file、doi、embedding_text 都已保留 |
| RAG 适配 | 256-word passage 更适合向量检索；PlantScience.ai 的 1500-token window 更偏 KG 抽取 |

## 4. 当前比 PlantScience.ai 差的地方

| 差距 | 影响 | 对标动作 |
| --- | --- | --- |
| 没有 OCR fallback | 扫描版 PDF/GROBID 失败时会断 | 后续加 OCR fallback，只处理 GROBID 失败文件 |
| strict peS2o 未完全启用 | pycld3/unigram 缺失，英文和低概率文本过滤只是 flag | 准备可选 strict resources，不默认硬拦 |
| metadata 仍弱 | 4/20 failed，主要是 title/abstract/残留问题 | 增强 metadata_override 和本地 title/abstract 检查 |
| 没有 1500-token extraction windows | 256-word chunk 适合 RAG，但关系抽取上下文偏短 | 新增 KG extraction windows，不替代 passage |
| 没有实体抽取 | 系统不知道 gene/species/phenotype 等实体 | 新增 entity_candidates |
| 没有关系抽取 | 不能形成知识边 | 新增 relation_candidates |
| 没有 KG | 不能 graph traversal | 先用 networkx 本地 KG |
| 没有 hybrid retrieval | 只能向量检索 | Chroma + KG node/edge 检索 |
| 没有证据型回答协议 | 回答 claim 和证据未结构化绑定 | 新增 evidence answer contract |
| 没有 continuous update | 新文献可处理，但不是 Agent 式持续更新 | 新增 paper registry / hash manifest |

## 5. 节省 token 的打法

不要把整篇 PDF 或全部 passages 塞给 LLM。

推荐上下文策略：

```text
第一层：只给 query + top passage IDs + node IDs
第二层：只展开必要 passage 的 1-3 句 evidence_text
第三层：只在需要验证时展开完整 256-word passage
```

KG 抽取策略：

```text
RAG embedding: 继续用 256-word OpenScholar passages
KG extraction: 另建 1500-token windows + 200-token overlap
LLM 输入: 只给 extraction window，不给整篇论文
输出: entity/relation JSON，保留 evidence_text 和 passage/window ID
```

回答策略：

```text
先检索，再压缩，再回答。
不要让 LLM 自己漫游数据库。
不要把 failed records 交给 LLM。
不要把 title/abstract/main_text 全量塞进 prompt。
```

Prompt 里优先传：

```text
question
top evidence snippets
paper title
doi
passage_id
kg edge triples
```

避免传：

```text
完整 PDF
完整 JSON record
完整 main_text
全部检索结果
无证据的 KG 节点
```

## 6. 对标实施计划

### P0：保住 PDF -> chunk 地基

不重写现有 pipeline。

只做维护：

```text
继续使用 scripts/run_pipeline.py
继续使用 scripts/pipeline/05_openscholar_passages.py
继续只让 index_ready records 入库
```

验收：

```text
block_words > 256 = 0
label_leak = 0
not_index_ready_passage = 0
embedding_text = title + "\n\n" + text
```

### P1：文献注册层

新增：

```text
scripts/agent/01_register_papers.py
data/agent/paper_registry.jsonl
```

用途：

```text
记录 paper_id、source_file、pdf_sha256、doi、title、year、index_ready。
```

对标：

```text
AutoSKG unique hash ID + source traceability
```

### P2：实体候选抽取

新增：

```text
scripts/agent/02_extract_entity_candidates.py
data/agent/entity_candidates.jsonl
```

第一版不用 LLM，先规则/词典：

```text
species
gene
protein
chemical
phenotype
tissue
method
disease
environmental_factor
```

每个 entity 必须有：

```text
entity_id
surface
normalized_name
entity_type
paper_id
passage_id
source_file
doi
evidence_text
extractor
confidence
```

### P3：关系候选抽取

新增：

```text
scripts/agent/03_extract_relation_candidates.py
data/agent/relation_candidates.jsonl
```

第一版关系：

```text
associated_with
affects
regulates
upregulates
downregulates
expressed_in
measured_by
observed_in_species
```

原则：

```text
没有 evidence_text，不入 relation。
没有 passage_id，不入 relation。
不让 LLM 脑补关系。
```

### P4：本地 KG

新增：

```text
scripts/agent/04_build_local_kg.py
data/agent/kg_nodes.jsonl
data/agent/kg_edges.jsonl
data/agent/local_kg.graphml
```

第一版后端：

```text
networkx
```

暂不上 Neo4j。

### P5：混合检索

新增：

```text
scripts/agent/05_hybrid_retrieve.py
```

流程：

```text
query
-> vector top-k passages
-> query entity match
-> KG 1-hop/2-hop expansion
-> evidence bundle
-> answer
```

### P6：证据型回答

新增：

```text
scripts/agent/06_answer_with_evidence.py
```

回答结构：

```json
{
  "answer": "...",
  "claims": [
    {
      "claim": "...",
      "supporting_passage_ids": [],
      "supporting_edge_ids": [],
      "confidence": "low|medium|high"
    }
  ],
  "citations": []
}
```

原则：

```text
有证据才回答。
证据弱就说不确定。
每个关键 claim 都绑定 passage 或 edge。
```

## 7. 第一阶段目录

计划新增：

```text
scripts/agent/
  01_register_papers.py
  02_extract_entity_candidates.py
  03_extract_relation_candidates.py
  04_build_local_kg.py
  05_hybrid_retrieve.py
  06_answer_with_evidence.py
  agent_config.py

data/agent/
  paper_registry.jsonl
  entity_candidates.jsonl
  relation_candidates.jsonl
  kg_nodes.jsonl
  kg_edges.jsonl
  local_kg.graphml
```

Git 规则：

```text
提交 scripts/agent/ 代码。
不提交 data/agent/ 生成产物，除非是极小测试 fixture。
```

## 8. 当前已落地

已新增：

```text
scripts/agent/01_register_papers.py
scripts/agent/02_extract_entity_candidates.py
scripts/agent/03_extract_relation_candidates.py
scripts/agent/04_build_local_kg.py
scripts/agent/05_hybrid_retrieve.py
scripts/agent/06_answer_with_evidence.py
scripts/agent/run_agent_pipeline.py
```

当前生成结果：

```text
paper_registry: 20 papers, 16 index_ready
entity_candidates: 1862 candidates
relation_candidates: 527 candidates
local_kg: 91 nodes, 217 edges
hybrid bundle: vector_evidence + graph_evidence + linked_passage_ids
evidence answer: claims + citations + limitations
```

当前质量约束：

```text
entity/relation 都必须带 paper_id + passage_id + evidence_text。
非 observed_in_species 关系不允许 species 误入主客体。
嵌套词关系已过滤，例如 DNA methylation -> methylation。
data/agent/ 为生成产物，不提交 Git。
```

下一步：

```text
继续提高 entity/relation 抽取质量：
- 引入领域词典
- 增加 negative patterns
- 增加人工抽样 QA
- 再考虑 LLM NER/关系抽取
```

一条命令跑 agent 层：

```powershell
conda activate rag
python scripts/agent/run_agent_pipeline.py "How does DNA methylation relate to zebrafish exposure?"
```
