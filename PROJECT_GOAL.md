# 鱼类 / 水产文献知识库与 GraphRAG 平台项目目标记录

本文档用于记录本项目的长期目标、对标对象、技术路线和阶段计划，作为后续开发、重构、评估和汇报的依据。

---

## 1. 项目名称

当前项目名称：

```text
鱼类文献本地知识库 RAG Demo
```

建议长期项目名称：

```text
鱼类 / 水产文献知识库与 GraphRAG 智能体平台
```

项目定位不是单纯做一个“PDF 聊天机器人”，而是以鱼类 / 水产文献为核心数据源，建设一个能够支持文献检索、单篇文献阅读、领域知识问答、知识图谱查询和证据追溯的科研辅助平台。

---

## 2. 对标对象与现实目标

本项目的直接参考对象是“油菜文献智能体 / Bn-iAgent”一类领域文献智能体，而不是泛泛地空谈大型 AI 平台。

油菜文献智能体的核心路线可以概括为：

```text
文献自动收集与筛选
-> 文献清洗与预处理
-> 文献 chunk 切分与 embedding
-> 向量知识库构建
-> 实体和关系抽取
-> 整合静态数据库
-> Neo4j 知识图谱
-> BN-LLM-KG-RAG
-> 文献智能问答与文献阅读工具
```

其已展示出的能力包括：

- 收集上万篇油菜遗传学相关文献。
- 将文献切分为 chunks，并通过 embedding 转化为高维向量存入向量数据库。
- 从知识向量库中识别关键实体及其关系。
- 整合人工收集的基因-性状关联列表、KEGG / GO 和 BnIR 基因注释信息。
- 使用 Neo4j 构建油菜知识图谱。
- 支持文献检索、基因功能问答、基因注释问答、品种特征问答。
- 支持单篇文献翻译、总结、提问、思维导图和 PPT 生成。
- 支持定期从 PubMed 等数据库自动获取新增文献并更新知识库。

本项目的现实目标是：

```text
在鱼类 / 水产方向复现并逐步超过上述路线。
```

其中，“超过”不是一开始就追求文献数量超过，而是技术路线要更清楚：

```text
GraphRAG 负责领域级、跨文献、跨实体的整体问答；
chunk + embedding RAG 负责单篇文献、局部证据、精确页码的问答；
知识图谱负责实体关系组织、路径查询、关系追溯和候选知识发现；
前后端平台负责稳定交互、任务管理和可视化展示。
```

---

## 3. 当前项目定位

当前项目是一个已经跑通基本闭环的鱼类 / 水产文献本地 RAG Demo。

现有主流程为：

```text
PDF 读取
-> 文本清洗
-> chunk 切分
-> embedding
-> ChromaDB 持久化
-> hybrid retrieval
-> DeepSeek 回答
-> 返回结构化引用来源
```

当前 demo 的价值在于：

- 验证了本地 PDF 文献可以被读取、清洗、切分和向量化。
- 验证了本地 embedding 模型可以和 ChromaDB 配合使用。
- 验证了 DeepSeek 可以基于检索片段生成中文回答。
- 验证了小规模鱼类 / 水产文献 RAG 的基本闭环。
- 初步实现了 vector search、BM25、关键词补召回和 query expansion 的混合检索。
- 初步实现了带 cited_sources 的结构化引用返回。

但当前系统仍然只是原型，不应被视为最终形态。主要限制包括：

- 文献规模较小，目前主要是约 20 篇 PDF 的 Demo。
- chunk 仍以固定长度切分为主，尚未充分利用论文结构。
- PDF 解析较粗糙，页眉页脚、参考文献噪音、断词、连字乱码等问题尚未彻底解决。
- metadata 不完整，缺少稳定的 title、DOI、year、journal、section、species 等字段。
- embedding 模型是通用多语言模型，不是专门面向生命科学 / 水产文献的模型。
- 检索虽然已加入 BM25 和关键词补召回，但尚缺成熟 reranker 和 graph retrieval。
- 当前尚未真正构建知识图谱，也没有系统性的实体抽取、关系抽取、实体标准化和人工评估流程。
- 当前 UI 使用 Streamlit，适合 Demo 展示，但不适合长期产品化和多任务管理。

因此，当前阶段的正确理解是：

```text
这是鱼类 / 水产领域文献智能体的最小 RAG 原型，
不是最终平台，也不是完整知识图谱系统。
```

---

## 4. 长期目标

本项目的长期目标是建设一个面向鱼类 / 水产遗传学、表观遗传学、养殖性状、病害与环境胁迫研究的文献知识库与 GraphRAG 智能体平台。

目标系统应从当前 RAG Demo 逐步演进为：

```text
鱼类 / 水产文献知识库
+ 单篇文献 RAG 阅读系统
+ 领域知识图谱
+ GraphRAG 跨文献问答系统
+ 文献自动更新与科研辅助平台
```

最终希望支持以下能力：

- 批量处理鱼类 / 水产相关文献。
- 对文献进行结构化解析、清洗、切分和入库。
- 支持基于单篇文献的翻译、总结、提问、解释和汇报材料生成。
- 支持基于整个文献库的领域问答、经典文献推荐、研究热点整理和证据追溯。
- 从文献中识别基因、蛋白、物种、性状、组织、通路、病原、环境因子、处理条件等关键实体。
- 从文献中抽取实体之间的关系，例如基因-性状关联、基因-通路关系、病原-宿主反应、环境因子-表型影响等。
- 整合 GO、KEGG、NCBI、Ensembl、UniProt、FishBase、QTL / GWAS 数据和人工整理表格。
- 使用 Neo4j 构建鱼类 / 水产领域知识图谱。
- 使用 GraphRAG 实现跨文献、跨实体、跨关系的问答。
- 返回可追溯证据，包括文献、页码、chunk_id、证据句、关系路径和置信度。

一句话概括：

```text
以 chunk + embedding RAG 解决“单篇文献读懂”的问题，
以 GraphRAG 解决“整个领域知识组织和跨文献问答”的问题。
```

---

## 5. 核心设计原则

### 5.1 不只做普通 RAG

普通 RAG 适合回答局部问题，但很难稳定回答领域级问题。例如：

```text
哪些基因与鱼类性别分化相关？
哪些通路反复出现在鱼类抗病研究中？
某个基因在不同鱼类中的功能是否一致？
某个性状有哪些经典文献和候选基因？
```

这些问题需要跨文献整合、实体归一化、关系组织和证据追溯，仅靠向量检索容易漏召回或答得很散。

因此，长期系统必须同时保留两条路线：

```text
单篇文献 RAG
领域 GraphRAG
```

---

### 5.2 单篇文献问题交给 chunk + embedding RAG

单篇文献 RAG 负责解决以下问题：

- 翻译这篇文献。
- 总结这篇文献。
- 解释某一段话。
- 对某一篇 PDF 提问。
- 生成该文献的推荐问题。
- 生成文献思维导图。
- 生成文献汇报 PPT 草稿。
- 查询某个结论在该文献哪一页、哪一段出现。

推荐流程：

```text
用户选择单篇 PDF
-> PDF 结构化解析
-> section-aware chunking
-> 单篇文献临时 / 独立向量索引
-> chunk 检索
-> LLM 生成答案
-> 返回页码、段落、证据句
```

这一部分的核心指标是：

```text
准确定位原文证据
减少幻觉
帮助用户快速读懂单篇论文
```

---

### 5.3 领域级问题交给 GraphRAG

GraphRAG 负责解决整个文献库层面的问题，例如：

- 某个性状有哪些相关基因？
- 某个基因关联了哪些性状、组织和通路？
- 某类环境胁迫影响了哪些表型和基因表达？
- 哪些文献是某个研究方向的经典文献？
- 某个病原感染后涉及哪些免疫通路？
- 某个水产物种的某类性状有哪些研究证据？
- 当前文献库中哪些关系证据最多，哪些关系证据不足？

推荐流程：

```text
用户问题
-> 识别实体和意图
-> 查询知识图谱中的相关实体与关系
-> 沿图谱扩展邻域
-> 找到支持关系的 evidence chunks
-> 同时进行 BM25 / vector 补充检索
-> reranker 重排证据
-> LLM 基于图谱路径和原文证据生成答案
-> 返回答案、图谱路径、证据句和文献来源
```

这一部分的核心指标是：

```text
跨文献整合能力
实体关系组织能力
证据可追溯能力
知识发现能力
```

---

## 6. 目标系统能力

### 6.1 文献检索与经典文献推荐

系统应支持用户询问：

```text
最近一个月有哪些鱼类表观遗传学相关文献？
鱼类性别分化研究中最经典的 10 篇文献是什么？
斑马鱼 DNA methylation 方向有哪些高价值文献？
嗜水气单胞菌感染鱼类的关键文献有哪些？
```

返回内容应包括：

- 文献标题。
- 年份、期刊、DOI。
- 研究对象和主题。
- 重要性评分。
- 推荐理由。
- 是否为新增文献。
- 与哪些实体或性状相关。

经典文献推荐可参考 PageRank、引用次数、图谱中心性、主题相关性和人工标注综合排序。

---

### 6.2 基因功能与基因注释问答

系统应支持用户询问：

```text
dnmt1 在鱼类中主要与什么甲基化过程相关？
cyp19a1a 与鱼类性别分化有什么关系？
某个鱼类基因对应的斑马鱼或拟南芥同源基因是什么？
某个基因参与哪些 GO / KEGG 通路？
某个基因在哪些组织中表达？
```

回答应同时来自：

- 文献证据。
- 公共数据库注释。
- 知识图谱关系。
- 原文 chunk。

返回内容应包括：

- 直接答案。
- 相关基因 / 同义名 / 同源基因。
- 相关性状、组织、通路。
- 支持文献和证据句。
- 图谱关系路径。

---

### 6.3 性状与病害机制问答

系统应支持用户询问：

```text
哪些基因与鱼类生长性状相关？
哪些基因与鱼类抗病性相关？
鱼类低氧胁迫涉及哪些通路？
嗜水气单胞菌感染后主要影响哪些免疫通路？
鱼类性别分化相关候选基因有哪些？
```

这一类问题不应只靠向量检索，而应优先走 GraphRAG：

```text
Trait / Disease / Pathogen / EnvironmentFactor
-> Graph neighbors
-> Gene / Pathway / Species / Tissue
-> Evidence chunks
-> 文献证据汇总
```

---

### 6.4 单篇文献阅读助手

系统应支持用户上传或选择单篇文献后进行：

- 全文翻译。
- 全文总结。
- 结构化摘要。
- 方法、结果、结论拆解。
- 自动推荐问题。
- 基于单篇文献的问答。
- 划词翻译、划词解释、划词提问。
- 笔记管理。
- 思维导图生成。
- PPT 大纲或汇报稿生成。

这一部分不需要复杂知识图谱，优先使用：

```text
PDF parser
+ section-aware chunks
+ single-paper vector index
+ LLM summarization
+ citation-aware QA
```

---

### 6.5 文献自动更新

系统应逐步支持定期从 PubMed、Semantic Scholar、Crossref、Web of Science 导出的题录文件或本地 PDF 文件夹中获取新增文献。

最小流程为：

```text
新增文献发现
-> metadata 抽取
-> 领域相关性筛选
-> PDF 下载 / 导入
-> 文本解析与清洗
-> chunk 入库
-> embedding 更新
-> 关键词索引更新
-> 实体关系抽取
-> 图谱更新
-> 新文献摘要生成
```

短期可以先做“本地文件夹增量更新”，不急于自动爬取所有数据库。

---

## 7. 目标数据对象

未来知识库需要支持以下核心对象：

```text
Paper
Chunk
Evidence
Gene
Protein
Trait
Species
Tissue
DevelopmentStage
Sex
Pathway
GO_term
KEGG_pathway
QTL
SNP
Disease
Pathogen
EnvironmentFactor
Treatment
Chemical
Method
Dataset
Journal
Author
```

这些对象不只是文本标签，而应尽量标准化为可追踪、可合并、可查询的实体。

例如：

```text
cyp19a
cyp19a1a
aromatase
gonadal aromatase
```

需要判断它们是同一基因的别名、不同亚型，还是不同物种中的同源基因。

再例如：

```text
zebrafish
Danio rerio
斑马鱼
```

应尽量映射到同一个标准物种实体。

---

## 8. 目标关系类型

未来知识图谱应支持但不限于以下关系：

```text
Paper HAS_CHUNK Chunk
Chunk MENTIONS Entity
Chunk SUPPORTS Evidence
Evidence ASSERTS Relation
Paper REPORTS Relation
Gene ASSOCIATED_WITH Trait
Gene REGULATES Trait
Gene EXPRESSED_IN Tissue
Gene PARTICIPATES_IN Pathway
Gene ANNOTATED_AS GO_term
Gene ANNOTATED_AS KEGG_pathway
Gene HAS_ORTHOLOG Gene
Species HAS_GENE Gene
Pathogen INFECTS Species
Pathogen INDUCES ImmuneResponse
Treatment AFFECTS Gene
Treatment AFFECTS Trait
EnvironmentFactor AFFECTS Trait
Chemical ALTERS Methylation
QTL ASSOCIATED_WITH Trait
SNP LOCATED_IN Gene
```

关系必须绑定证据链，不能只保存孤立三元组。

每条关系至少应包含：

```text
relation_id
head_entity
head_type
relation_type
tail_entity
tail_type
confidence
evidence_sentence
paper_id
source_file
chunk_id
page_number
section
extraction_method
model_version
human_verified
```

---

## 9. 双 RAG 架构

本项目的核心架构应明确区分两种 RAG。

### 9.1 Single-Paper RAG

用途：

```text
单篇文献阅读、翻译、总结、问答、划词解释、PPT 生成
```

特点：

- 范围小。
- 证据定位精确。
- 适合页码级引用。
- 不需要图谱即可完成。
- 重点是读懂一篇文章。

技术路线：

```text
PDF parser
-> section-aware chunking
-> single-paper vector index
-> local retrieval
-> LLM answer
-> page / section / chunk citation
```

---

### 9.2 Domain GraphRAG

用途：

```text
跨文献知识问答、基因功能查询、性状机制总结、经典文献推荐、知识发现
```

特点：

- 范围大。
- 需要实体标准化。
- 需要关系网络。
- 适合跨文献综合。
- 重点是组织整个领域知识。

技术路线：

```text
entity recognition
-> entity normalization
-> relation extraction
-> evidence linking
-> Neo4j graph storage
-> graph traversal
-> vector / BM25 evidence retrieval
-> reranker
-> LLM answer
```

---

### 9.3 Query Router

系统需要一个问题路由模块，用于判断用户问题走哪条路径。

示例：

```text
“总结这篇论文” -> Single-Paper RAG
“解释这段话” -> Single-Paper RAG
“这篇文章的创新点是什么” -> Single-Paper RAG
“哪些基因与鱼类性别分化相关” -> Domain GraphRAG
“某个基因在哪些鱼类中被报道过” -> Domain GraphRAG
“最近一个月有哪些新文献” -> Literature Update + Domain Retrieval
“推荐 10 篇经典文献” -> Graph ranking + Literature Retrieval
```

---

## 10. 目标技术架构

短期系统可以继续使用当前架构：

```text
Streamlit
+ ChromaDB
+ local embedding model
+ DeepSeek API
+ local PDF folder
```

但长期平台建议演进为：

```text
Frontend:
  React / Vue / Next.js

Backend API:
  FastAPI

Task Queue:
  Celery / RQ / Dramatiq

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

Parser:
  GROBID / PyMuPDF / ScienceParse-like parser
```

选择原则：

```text
短期先稳定，不追求炫技；
中期逐步工程化，不推倒重写；
长期前后端分离，支持任务管理、图谱可视化和多用户使用。
```

---

## 11. 文献处理目标

未来 PDF 处理不应只依赖简单文本读取。

应逐步实现：

- 标题、作者、摘要、DOI、期刊、年份抽取。
- 章节识别，例如 Abstract、Introduction、Methods、Results、Discussion。
- 页码、段落、chunk_id 保留。
- 表格标题和图注解析。
- 参考文献区域识别与降权。
- 页眉页脚清理。
- 断词和连字乱码修复。
- 文献 metadata manifest 维护。

每个 chunk 应尽量保留：

```text
chunk_id
paper_id
source_file
title
DOI
year
journal
section
page_number
paragraph_index
char_offset
chunk_text
evidence_type
```

---

## 12. 检索目标

未来检索应从单一路径演进为多路融合：

```text
BM25 / keyword retrieval
+ vector retrieval
+ graph retrieval
+ reranker
-> evidence selection
-> LLM answer
```

检索质量的核心不是返回更多片段，而是返回能支撑回答的正确证据。

需要重点改进：

- query rewrite / bilingual expansion。
- entity-aware retrieval。
- exact entity match 加权。
- section-aware rerank。
- references chunk 降权。
- domain reranker。
- graph-neighborhood retrieval。
- answer-grounded citation verification。

---

## 13. 知识图谱目标

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
(:Disease)
(:Pathogen)
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
(:Gene)-[:EXPRESSED_IN]->(:Tissue)
(:Gene)-[:PARTICIPATES_IN]->(:Pathway)
(:Gene)-[:ANNOTATED_AS]->(:GO)
(:Gene)-[:ANNOTATED_AS]->(:KEGG)
(:Species)-[:HAS_GENE]->(:Gene)
(:Pathogen)-[:INFECTS]->(:Species)
(:Treatment)-[:AFFECTS]->(:Gene)
(:EnvironmentFactor)-[:AFFECTS]->(:Trait)
```

所有抽取关系都应能追溯到原始文献、chunk 和 evidence sentence。

---

## 14. 评估目标

没有评估，图谱规模越大，错误也会越大。

未来至少需要四类评估。

### 14.1 Retrieval Evaluation

用于评估检索是否命中正确文献和 chunk。

指标：

```text
Hit@K
MRR
Recall@K
source relevance
evidence relevance
```

---

### 14.2 Single-Paper QA Evaluation

用于评估单篇文献问答是否准确。

指标：

```text
answer correctness
citation correctness
page hit rate
unsupported claim rate
```

---

### 14.3 Entity Extraction Evaluation

用于评估实体抽取质量。

指标：

```text
Precision
Recall
F1
entity normalization accuracy
```

---

### 14.4 Relation Extraction Evaluation

用于评估关系抽取质量。

指标：

```text
Precision
Recall
F1
evidence correctness
human verification pass rate
```

评估原则：

```text
宁可少抽，也不要大量错误关系入图；
每一条重要关系都应能追溯到证据句；
核心关系需要人工抽检。
```

---

## 15. 阶段路线

### 阶段 0：当前 RAG Demo

规模：

```text
约 20 篇 PDF
本地 embedding
ChromaDB
Streamlit
DeepSeek 回答
```

目标：

- 跑通 RAG 闭环。
- 验证小规模文献问答可行。
- 初步实现 hybrid retrieval。
- 初步返回 cited_sources。

当前状态：

```text
已基本完成，但回答质量仍不稳定。
```

---

### 阶段 1：稳定单篇文献 RAG 与文献库 RAG

规模：

```text
20 到 100 篇文献
```

目标：

- 改进 PDF 清洗。
- 改进 section-aware chunking。
- 建立 papers manifest。
- 完善 title、DOI、year、journal、section、species 等 metadata。
- 建立小型 RAG eval 集。
- 支持单篇文献总结、提问和证据定位。
- 支持全库基础问答。

本阶段重点：

```text
先让“能答的问题答得稳”，不要急于堆功能。
```

---

### 阶段 2：工程化 RAG 与前后端雏形

规模：

```text
100 到 500 篇文献
```

目标：

- 使用 FastAPI 封装后端接口。
- 使用 React / Vue / Next.js 搭建更稳定的前端雏形。
- 使用 PostgreSQL 管理文献 metadata。
- 使用 Qdrant / Milvus 替代或补充 Chroma。
- 引入 OpenSearch / Elasticsearch。
- 引入 reranker。
- 增强增量更新流程。
- 支持文献列表、问答历史、证据查看、单篇文献阅读。

本阶段重点：

```text
把 Demo 变成可维护的工程系统。
```

---

### 阶段 3：小型知识图谱与 GraphRAG 原型

规模：

```text
500 到 2000 篇文献
```

目标：

- 定义实体 schema 和关系 schema。
- 抽取 10 到 15 类核心实体。
- 抽取 10 到 20 类核心关系。
- 引入 Neo4j。
- 每条关系绑定 evidence。
- 建立人工抽检流程。
- 支持图谱查询和证据追溯。
- 实现 GraphRAG 原型。

本阶段重点：

```text
不追求图谱最大，先追求关系真实、证据完整、查询可用。
```

---

### 阶段 4：鱼类 / 水产领域文献智能体平台

规模：

```text
数千到上万篇文献
```

目标：

- 自动化文献入库。
- 大规模实体标准化。
- 大规模关系抽取。
- 整合 GO、KEGG、NCBI、Ensembl、UniProt、FishBase 等公共数据。
- 构建鱼类 / 水产领域知识图谱。
- 支持 GraphRAG。
- 支持文献检索、基因功能、基因注释、性状机制、病害机制、单篇文献阅读等场景。
- 支持图谱分析、经典文献推荐和知识空白发现。

本阶段目标：

```text
形成对标油菜文献智能体的鱼类 / 水产文献智能体平台。
```

---

## 16. 当前最重要的开发方向

接下来不应继续空泛扩展，而应围绕以下主线推进：

```text
1. 稳定当前 RAG 闭环
2. 建立单篇文献 RAG 能力
3. 建立 papers manifest 和 metadata 管理
4. 建立小型评估集
5. 设计知识图谱 schema
6. 手工构造少量 evidence-based relation demo
7. 再逐步实现实体抽取、关系抽取和 Neo4j 入图
```

短期最关键的不是立刻做一个巨大图谱，而是先把数据格式、证据链和评估方式固定下来。

---

## 17. 当前不建议立即做的事情

当前阶段不建议：

- 盲目爬取几万篇 PDF。
- 直接上复杂大前端。
- 直接堆 Neo4j 但没有 schema。
- 直接抽大量关系但没有人工评估。
- 直接训练领域大模型。
- 只追求页面好看而忽略证据质量。
- 把 GraphRAG 当成普通 RAG 的装饰词。

当前应优先保证：

```text
文献能稳定入库；
证据能稳定检索；
回答能引用来源；
实体和关系能追溯到原文；
系统效果能被评估。
```

---

## 18. 一句话总结

本项目的目标不是只做一个简单的 PDF 问答 Demo，而是参考油菜文献智能体的路线，在鱼类 / 水产方向构建一个更清晰的“双 RAG”文献智能体平台：

```text
用 chunk + embedding RAG 解决单篇文献阅读和局部证据问答，
用知识图谱 + GraphRAG 解决跨文献、跨实体、跨关系的领域知识问答，
最终形成可更新、可追溯、可评估的鱼类 / 水产文献知识库与科研辅助平台。
```
