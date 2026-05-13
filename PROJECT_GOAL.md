# 鱼类文献知识库项目目标记录

本文档用于记录本项目的长期目标和阶段性路线，作为后续开发、重构和评估的依据。

## 当前项目定位

当前项目是一个“鱼类/水产文献本地知识库 RAG Demo”。

现有主流程为：

```text
PDF 读取 -> 文本清洗 -> chunk 切分 -> embedding -> ChromaDB 持久化 -> 检索 -> DeepSeek 回答 -> 返回引用来源
```

当前 demo 的价值在于：

- 验证了本地 PDF 文献可以被读取、切分、向量化和检索。
- 验证了本地 embedding 模型可以和 ChromaDB 配合使用。
- 验证了 DeepSeek 可以基于检索片段生成带引用的中文回答。
- 验证了小规模鱼类/水产文献 RAG 的基本闭环。

但当前系统仍是原型，不应被视为最终形态。其主要限制包括：

- chunk 仍以固定长度切分为主，尚未充分利用论文结构。
- embedding 模型是通用多语言模型，不是专门的生命科学/生物医学文献模型。
- 检索主要依赖向量检索、BM25 和关键词补召回，尚无成熟 reranker。
- PDF 解析、页眉页脚、参考文献噪音、连字乱码等问题仍未彻底解决。
- metadata 仍不完整，缺少稳定的 title、DOI、year、journal、section、species 等字段。
- 当前 UI 使用 Streamlit，适合 demo，不适合长期产品化和大规模任务管理。
- 当前还没有真正构建知识图谱，也没有系统性的实体抽取、关系抽取和人工评估流程。

## 长期目标

本项目的长期目标不是停留在简单 RAG 问答，而是建设一个面向鱼类/水产遗传学、表观遗传学和养殖性状研究的领域知识库。

目标系统应逐步演进为：

```text
文献知识工程系统
+ 实体关系抽取系统
+ 知识图谱系统
+ 图谱增强 RAG / GraphRAG 系统
```

最终希望实现类似以下能力：

- 批量处理大规模鱼类/水产相关文献。
- 将文献切分为结构化片段，并保留精确证据链。
- 从文献中识别基因、蛋白、性状、物种、组织、通路、处理条件、环境因子等关键实体。
- 从文献中抽取实体之间的关系，例如基因-性状关联、环境因子-表型影响、基因-通路关系等。
- 整合公共数据库和人工整理数据集，例如 GO、KEGG、NCBI、Ensembl、FishBase、QTL/GWAS 数据、人工整理的基因-性状关联表。
- 使用 Neo4j 构建鱼类/水产领域知识图谱。
- 支持图谱查询、证据追溯、文献问答和图谱增强推理。

## 参考目标形态

目标系统可以参考大型作物遗传学知识图谱项目的思路：

- 将上万篇领域文献切分为 chunks。
- 使用 embedding 模型将 chunks 转化为高维向量并存入向量数据库。
- 从知识向量库中识别关键实体及其关系。
- 整合静态数据集，包括人工收集的基因-性状关联列表、KEGG/GO 数据和领域数据库中的基因注释信息。
- 使用 Neo4j 将实体和关系构建为知识图谱。
- 对基因、蛋白质、性状等关键实体进行抽取和标准化。
- 使用大型语言模型辅助文本挖掘，构建大规模实体关系网络。
- 分析知识图谱的 degree 分布、核心实体、关键关系和潜在知识空白。

本项目希望在鱼类/水产方向逐步实现类似能力，但不要求一步到位。

## 目标数据对象

未来知识库需要支持的核心对象包括：

- Paper
- Chunk
- Evidence
- Gene
- Protein
- Trait
- Species
- Tissue
- DevelopmentStage
- Sex
- EnvironmentFactor
- Treatment
- Chemical
- Pathway
- GO term
- KEGG pathway
- QTL
- SNP
- Method
- Dataset
- Journal
- Author

这些对象不只是文本标签，而应尽量标准化为可追踪、可合并、可查询的实体。

例如：

```text
cyp19a
cyp19a1a
aromatase
gonadal aromatase
```

需要判断它们是同一实体的别名、同源基因，还是不同物种/亚型下的不同实体。

## 目标关系类型

未来知识图谱应支持的关系包括但不限于：

- Gene associated_with Trait
- Gene regulates Trait
- Gene expressed_in Tissue
- Gene participates_in Pathway
- Gene annotated_as GO_term
- Gene annotated_as KEGG_pathway
- Gene has_ortholog Gene
- Treatment affects Gene
- Treatment affects Trait
- EnvironmentFactor induces Phenotype
- Chemical alters Methylation
- Species has_gene Gene
- Paper reports Relation
- Chunk supports Relation
- Evidence extracted_from Chunk

关系必须绑定证据链，不能只保存孤立三元组。

每条关系至少应包含：

- head_entity
- relation_type
- tail_entity
- confidence
- evidence_sentence
- paper_id
- chunk_id
- page_number
- section
- extraction_method
- model_version
- human_verified

## 目标系统架构

当前 Streamlit + Chroma 适合 demo，但长期系统应逐步升级为更完整的工程架构。

建议目标架构：

```text
Frontend:
  React / Vue

Backend API:
  FastAPI

Task Queue:
  Celery / RQ

Relational Metadata DB:
  PostgreSQL

Vector DB:
  Qdrant / Milvus / Weaviate

Keyword Search:
  OpenSearch / Elasticsearch

Graph DB:
  Neo4j

Object Storage:
  Local filesystem / MinIO

LLM Service:
  DeepSeek / OpenAI-compatible API / local model
```

后台任务应包括：

- PDF ingestion
- metadata extraction
- structured parsing
- section-aware chunking
- embedding
- vector index update
- keyword index update
- entity extraction
- entity normalization
- relation extraction
- evidence verification
- Neo4j import
- evaluation

## 文献处理目标

未来 PDF 处理不应只依赖简单文本读取。

建议引入：

- GROBID：解析论文结构、标题、作者、摘要、章节、参考文献。
- DOI / Crossref / PubMed 等 metadata 补全。
- Section-aware chunking。
- Table caption 和 figure caption 解析。
- References 区域识别和降权/过滤。
- 页眉页脚、版权声明、断词、连字乱码清洗。

每个 chunk 应尽量保留：

- paper_id
- source_file
- title
- DOI
- year
- journal
- section
- page_number
- paragraph_index
- char_offset
- chunk_id
- evidence_type

## 检索目标

未来检索应从单一路径演进为多路融合：

```text
BM25 / keyword 检索
+ vector 检索
+ graph 检索
+ reranker 重排
-> evidence selection
-> LLM answer
```

检索质量的核心不是“返回更多片段”，而是“返回能支撑回答的正确证据”。

需要重点改进：

- query rewrite / bilingual expansion
- entity-aware retrieval
- exact entity match 加权
- section-aware rerank
- references chunk 降权
- domain reranker
- graph-neighborhood retrieval
- answer-grounded citation verification

## 知识图谱目标

Neo4j 不只是数据库替换，而是知识组织方式的升级。

最小图模型可以从以下节点开始：

```text
(:Paper)
(:Chunk)
(:Evidence)
(:Gene)
(:Protein)
(:Trait)
(:Species)
(:Tissue)
(:Pathway)
(:GO)
(:KEGG)
(:Treatment)
(:EnvironmentFactor)
```

基础关系可以从以下类型开始：

```text
(:Paper)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Gene)
(:Chunk)-[:MENTIONS]->(:Trait)
(:Chunk)-[:SUPPORTS]->(:Evidence)
(:Evidence)-[:ASSERTS]->(:Relation)
(:Gene)-[:ASSOCIATED_WITH]->(:Trait)
(:Gene)-[:PARTICIPATES_IN]->(:Pathway)
(:Gene)-[:ANNOTATED_AS]->(:GO)
(:Gene)-[:ANNOTATED_AS]->(:KEGG)
```

所有抽取关系都应能追溯到原始文献和 chunk。

## 评估目标

没有评估，图谱规模越大，错误也会越大。

未来至少需要三套评估：

### Retrieval Evaluation

用于评估检索是否命中正确文献和 chunk。

指标：

- Hit@K
- MRR
- Recall@K
- source relevance

### Entity Extraction Evaluation

用于评估实体抽取质量。

指标：

- Precision
- Recall
- F1
- entity normalization accuracy

### Relation Extraction Evaluation

用于评估关系抽取质量。

指标：

- Precision
- Recall
- F1
- evidence correctness
- human verification pass rate

目标不是让系统“看起来能答”，而是让每一步都有可衡量的质量。

## 阶段路线

### 阶段 1：当前 RAG Demo

规模：

- 约 20 篇 PDF
- 本地 embedding
- ChromaDB
- Streamlit
- DeepSeek 回答

目标：

- 跑通 RAG 闭环。
- 验证小规模文献问答可行。

### 阶段 2：工程化 RAG

规模：

- 100 到 500 篇文献

目标：

- 引入 GROBID 或更稳定的论文解析。
- 改进 section-aware chunking。
- 建立 papers manifest。
- 使用 Qdrant / Milvus 替代或补充 Chroma。
- 引入 OpenSearch / Elasticsearch。
- 使用 FastAPI 替代 Streamlit 作为后端。
- 加入 reranker。
- 建立 retrieval evaluation。

### 阶段 3：小型知识图谱

规模：

- 500 到 2000 篇文献

目标：

- 定义实体 schema 和关系 schema。
- 抽取 10 到 15 类实体。
- 抽取 10 到 20 类关系。
- 引入 Neo4j。
- 每条关系绑定 evidence。
- 建立人工抽检流程。
- 支持图谱查询和证据追溯。

### 阶段 4：领域知识库

规模：

- 数千到上万篇文献。

目标：

- 自动化文献入库。
- 大规模实体标准化。
- 大规模关系抽取。
- 整合 GO、KEGG、NCBI、Ensembl、FishBase 等公共数据。
- 构建鱼类/水产领域知识图谱。
- 支持 GraphRAG。
- 支持图谱分析和知识发现。

## 当前结论

当前项目是合理的小型 demo，但距离大型领域知识图谱系统还有明显差距。

后续最关键的升级方向不是简单替换数据库，而是：

```text
结构化解析论文
-> 标准化实体
-> 可验证关系抽取
-> 带证据链入图
-> 图谱 + 向量 + 关键词混合检索
-> 可评估的问答和知识发现
```

Neo4j 是承载知识图谱的基础设施，但真正的核心在于：

- 实体 schema
- 关系 schema
- 实体标准化
- 证据链
- 人工评估
- 持续迭代的数据管线

本文件作为后续开发目标记录，后续所有较大改动应尽量围绕该路线推进。
