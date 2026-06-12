# GraphRAG 图结构优化设计

## 目标

X5 的目标不是引入复杂图数据库，而是在当前本地项目中做一个可解释的 GraphRAG-lite：

```text
原始检索 seed chunks
-> 根据文档图关系扩展候选 chunk
-> 给每个扩展 chunk 附上关系分数和原因
-> 交给 MiniRanker 重新排序
-> 页面展示为什么某个 chunk 被扩展出来
```

## 图节点

当前图结构包含四类节点：

```text
file    文件节点
page    页面节点
chunk   文本片段节点
concept 知识点节点
```

每个 `chunk` 节点保留：

```text
chunk_id
file_name
page
chapter
concepts
```

## 图边

当前支持这些边：

```text
contains_page    file -> page
contains_chunk   page -> chunk
mentions_concept chunk -> concept
same_page        chunk 与 seed chunk 同页
same_chapter     chunk 与 seed chunk 同章节
adjacent         chunk 与 seed chunk 在同文件中相邻
shared_concept   chunk 与 seed chunk 共享知识点
```

用于扩展候选的主要关系是：

```text
same_page
same_chapter
adjacent
shared_concept
```

## 关系分数

当前默认分数：

```text
same_page        0.30
same_chapter     0.20
adjacent         0.25
shared_concept   0.40 + 0.10 * 共享知识点数量
```

如果一个候选 chunk 与多个 seed chunk 有关系，取最高关系组合分数作为 `graph_score`。

## 页面解释

问答页新增两类展示：

```text
检索与重排序结果
GraphRAG 扩展解释
```

`检索与重排序结果` 会显示：

```text
来源：原始检索 / GraphRAG扩展
dense
bm25
graph
ranker
GraphRAG原因
```

`GraphRAG 扩展解释` 会显示：

```text
扩展chunk
种子chunk
关系
分数
原因
文件
页码
```

这样汇报时可以解释：

```text
这个片段不是向量检索直接命中的，
它是因为与原始命中片段同页、同章节、相邻或共享知识点，
所以被 GraphRAG 扩展进候选集。
```

## 验证

已补充测试：

```text
tests/test_graph_store.py
```

覆盖内容：

```text
同页扩展
共享知识点扩展
相邻 chunk 扩展
稳定排序
非法 hops 检查
图节点和图边构造
扩展解释关系
```
