# 组员数据贡献目录

每个组员在这里建立自己的文件夹：

```text
data_contrib/姓名/
  raw/
  metadata.csv
  questions.csv
```

要求：

```text
raw/ 放原始资料文件
metadata.csv 记录每个资料文件的来源、主题和是否可提交
questions.csv 记录用于测试检索效果的问题
```

如果资料不能公开上传 GitHub，在 `metadata.csv` 的 `can_commit` 写 `no`，并先不要把原文件提交到仓库。

全组共享、确认可以提交的资料也可以直接放到：

```text
data/raw/
```
