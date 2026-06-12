# CourseMind 数据收集指导

本文档给刚接触项目的组员使用。当前知识库主题统一为：

```text
深度学习课程知识检索与学习助手
```

## 范围边界

只收集“课程知识内容”，不收集项目管理规则。

纳入知识库：

```text
神经网络基础
反向传播
损失函数
优化器
过拟合与正则化
CNN
RNN
LSTM
GRU
Transformer
注意力机制
自编码器
GAN
深度强化学习 / DRL
DQN
策略梯度
Actor-Critic
模型训练
模型调参
实验结果分析
```

不纳入知识库：

```text
课程项目说明
项目要求
汇报要求
评分标准
成员贡献要求
RAG
FAISS
GraphRAG
MiniRanker
Bandit 推荐
OCR
Streamlit
```

说明：课程项目说明、评分标准、汇报要求只用于我们做项目管理和 PPT，不作为组员的数据收集主题。RAG、FAISS、GraphRAG、Bandit 等是 CourseMind 的系统实现技术，也不作为课程知识库主题。

## Git 提交规则

仓库现在允许提交 `data/raw/` 下面的原始资料。组员如果已经把资料放到 `data/raw/`，可以正常 `git add` 和提交。

推荐提交位置：

```text
data/raw/
data_contrib/姓名/
```

使用建议：

```text
全组共享、确认可以提交的资料 -> 放 data/raw/
每个组员整理的一整包资料 -> 放 data_contrib/姓名/
系统生成的 chunks、FAISS 索引 -> 不提交
```

注意：既然 `data/raw/` 不再被忽略，就不要把不能公开上传的老师课件、版权不清楚的 PDF、个人信息、账号密码、API Key 放进去。不确定能不能提交的资料，先不要进仓库，并在 `metadata.csv` 的 `can_commit` 写 `no`。

## 每个人要交什么

每个人提交：

```text
资料文件 + metadata.csv + questions.csv
```

没有 `questions.csv` 的资料，暂时不算完整贡献。

## 推荐提交结构

每个成员建一个自己的文件夹：

```text
data_contrib/姓名/
  raw/
    CNN_卷积神经网络_张三.md
    Transformer_注意力机制_张三.txt
    LSTM_长短期记忆网络_张三.pdf
  metadata.csv
  questions.csv
```

文件命名建议：

```text
主题_具体内容_姓名.md
主题_具体内容_姓名.txt
主题_具体内容_姓名.pdf
```

不要使用：

```text
新建文本文档.txt
资料.pdf
111.md
未命名.pdf
```

## 优先收什么资料

优先收集：

1. 中文深度学习课程讲义 PDF
2. 深度学习 PPT 导出的 PDF
3. 课堂笔记 Markdown / TXT
4. 教材章节摘要
5. 个人整理的概念笔记
6. 公开资料的中文摘要或个人整理版

推荐主题：

```text
神经网络基础
反向传播
损失函数
优化器
过拟合与正则化
CNN
卷积、池化、特征图
RNN
LSTM
GRU
Transformer
Self-Attention
多头注意力
位置编码
自编码器
GAN
深度强化学习
DQN
策略梯度
Actor-Critic
模型训练与调参
实验结果分析
```

## 暂时不要收什么

先不要提交：

1. 扫描版 PDF
2. 图片截图
3. 只有图片、没有可复制文字的 PDF
4. 视频、音频
5. 版权不清楚的大段教材原文
6. 内容太短、没有检索价值的文件
7. 项目说明、评分标准、汇报规则
8. 和深度学习课程无关的系统技术教程

判断 PDF 能不能用：

```text
打开 PDF
选中一段文字
复制到记事本
如果能复制出正常中文，优先收
如果复制不出来，先不要收
```

## metadata.csv 怎么写

`metadata.csv` 用来说明每个资料文件是什么、从哪里来、能不能提交到 Git。

字段：

```csv
file_name,title,source,type,topic,can_commit,notes
```

示例：

```csv
file_name,title,source,type,topic,can_commit,notes
CNN_卷积神经网络_张三.md,CNN 卷积神经网络笔记,个人整理,markdown,CNN,yes,可提交
Transformer_注意力机制_张三.txt,Transformer 注意力机制笔记,个人整理,txt,Transformer,yes,可提交
LSTM_长短期记忆网络_张三.pdf,LSTM 长短期记忆网络讲义,课程讲义,pdf,LSTM,no,只本地使用
```

字段说明：

```text
file_name   : raw/ 下面的文件名，必须写准确
title       : 资料标题
source      : 来源，例如个人整理、课程讲义、公开资料摘要
type        : pdf / txt / markdown
topic       : 主题，例如 CNN、Transformer、LSTM、DRL
can_commit  : yes 或 no。不能公开上传 Git 的写 no
notes       : 其他说明
```

## questions.csv 怎么写

`questions.csv` 用来做检索和问答测试。每条问题都应该能在资料里找到依据。

字段：

```csv
query,expected_file,expected_topic,query_type,answer_keywords
```

示例：

```csv
query,expected_file,expected_topic,query_type,answer_keywords
卷积神经网络为什么适合图像任务？,CNN_卷积神经网络_张三.md,CNN,fact,"局部连接;权值共享;特征图"
LSTM 如何缓解 RNN 的长期依赖问题？,LSTM_长短期记忆网络_张三.pdf,LSTM,fact,"门控;遗忘门;记忆单元"
Transformer 的注意力机制有什么作用？,Transformer_注意力机制_张三.txt,Transformer,fact,"Query;Key;Value;权重"
CNN 和 RNN 的主要区别是什么？,CNN_卷积神经网络_张三.md,CNN,compare,"图像;序列;卷积;循环"
这个资料里有没有讲世界杯？,CNN_卷积神经网络_张三.md,资料外,out_of_scope,""
```

字段说明：

```text
query           : 用户可能会问的问题
expected_file   : 期望命中的资料文件
expected_topic  : 问题所属主题
query_type      : fact / summary / compare / quiz / out_of_scope
answer_keywords : 答案里应该出现的关键词，用英文分号 ; 分隔
```

## 每个人最低交付

每个人至少提交：

```text
3 个中文文本型资料文件
10 个测试问题
1 个资料外问题
1 个 metadata.csv
1 个 questions.csv
```

其中 10 个测试问题建议这样分：

```text
6 个事实问答
2 个对比问题
1 个总结问题
1 个资料外拒答问题
```

资料外问题示例：

```text
这个资料里有没有讲 YOLOv10？
这个资料里有没有讲股票预测？
这个资料里有没有讲世界杯？
```

这些问题用于测试系统的拒答机制。

## 好问题和坏问题

好问题：

```text
CNN 中卷积核的作用是什么？
LSTM 的遗忘门有什么作用？
Transformer 为什么需要位置编码？
DQN 和普通 Q-learning 有什么区别？
Actor-Critic 中 actor 和 critic 分别负责什么？
```

坏问题：

```text
介绍一下这个。
这个是什么？
说说看。
帮我总结。
这个资料好不好？
```

坏问题太泛，不方便判断系统有没有答对。

## 资料质量要求

资料要满足：

1. 中文内容为主
2. 能复制文字
3. 内容和深度学习课程知识相关
4. 文件名清晰
5. 不要只有一两句话
6. 不要提交 API Key、账号密码、私人信息
7. 不确定版权能不能公开的资料，`can_commit` 写 `no`

## 为什么必须写问题

CourseMind 不只是存资料，还要验证检索效果。`questions.csv` 后面可以直接变成实验评测集：

```text
问题
-> 系统检索 Top-K
-> 检查有没有命中 expected_file / expected_topic
-> 统计 Recall@5
-> 比较 BM25、FAISS、GraphRAG、MiniRanker 的效果
```

所以只交资料不交问题，后面仍然无法证明系统有效。

## 分工建议

如果不知道自己该收什么，可以按主题分：

```text
成员 A：神经网络基础、反向传播、优化器
成员 B：CNN、图像任务、卷积结构
成员 C：RNN、LSTM、GRU、序列建模
成员 D：Transformer、注意力机制、位置编码
成员 E：深度强化学习、DQN、策略梯度
成员 F：自编码器、GAN、模型训练与调参
```

如果人数少，可以合并：

```text
1. 神经网络基础 + CNN
2. RNN/LSTM/GRU + Transformer
3. 深度强化学习
4. 自编码器/GAN + 模型训练与调参
```

## 最终提交前自查

提交前检查：

```text
[ ] raw/ 下面有至少 3 个资料文件
[ ] metadata.csv 每个文件都有一行记录
[ ] questions.csv 至少 10 个问题
[ ] 至少 1 个 out_of_scope 问题
[ ] PDF 可以复制正常中文
[ ] 文件名清楚
[ ] can_commit 已经填写 yes/no
[ ] 没有提交私人信息或 API Key
```

## 给组员的提醒

这不是单纯找资料任务。每个人的有效贡献必须能被入库验证：

```text
资料能被解析
问题能被检索
答案能被评测
贡献能被记录
```

如果临时换人接手，只要看到你的 `metadata.csv` 和 `questions.csv`，就应该知道你交了什么、资料能干什么、怎么测试。
