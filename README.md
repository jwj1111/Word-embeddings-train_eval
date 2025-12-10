# Word Embedding Training & Evaluation

# 词嵌入模型训练与评测

本项目旨在探索英文词嵌入（Word Embeddings）模型的训练与评估。通过使用 OpenSubtitles 语料库训练基于 **Skip-Gram** 和 **CBOW** 的 Word2Vec 模型，对比不同超参数（窗口大小、向量维度、训练轮数）对模型性能的影响，并将自训练模型与业界主流的预训练模型（Word2Vec, GloVe, FastText）进行详细的对比评测。

## 项目目标

1.  **模型训练**：基于 Skip-Gram 和 CBOW 架构训练 36 个不同参数组合的 Word2Vec 模型。
2.  **超参数分析**：评估窗口大小、词向量维度等参数对语义相似度和类比推理任务的影响。
3.  **对比评测**：将自训练模型与 Google News Word2Vec、GloVe、FastText 等预训练模型进行性能对比。

## 数据集与预处理

  * **原始语料**：[可疑链接已删除] (英文单语语料)。
  * **预处理工具**：`spaCy 3.7.5` (en\_core\_web\_sm)。
  * **处理流程**：
      * 按句载入语料。
      * 去除空白字符、标点符号。
      * 统一转换为小写形式。
  * **训练规模**：控制在约 **3000 万 (30M)** 个有效 Token。
  * **输出文件**：`code_data/text_processed/en_processed.txt`

## 🛠️ 实验方法

### 1\. 模型训练 (Gensim)

使用 `gensim 4.3.2` 进行模型训练，采用了网格搜索（Grid Search）的方式探索以下参数组合：

| 参数 | 说明 | 设置值 |
| :--- | :--- | :--- |
| **Model Arch** | 模型架构 | Skip-Gram (sg=1), CBOW (sg=0) |
| **Window** | 上下文窗口大小 | 5, 7, 9 |
| **Vector Size** | 词向量维度 | 200, 300 |
| **Epochs** | 训练轮数 | 5, 7, 10 |
| **Min Count** | 最小词频 | 5 |

### 2\. 预训练模型对比

本项目引入了以下预训练模型作为基准（Baseline）：

  * `word2vec-google-news-300`
  * `glove-wiki-gigaword-300`
  * `fasttext-wiki-news-subwords-300`

## 评测任务与指标

### 任务一：语义相似度 (Semantic Similarity)

测试模型判断两个词之间语义相似程度的能力。

  * **数据集**：WordSim-353, SimLex-999。
  * **评价指标**：
      * Pearson 相关系数 ($r$)
      * Spearman 等级相关系数 ($\rho$)

### 任务二：类比推理 (Analogy Reasoning)

测试模型进行 "A is to B as C is to ?" 推理的能力。

  * **数据集**：Google Analogy Test Set。
  * **子任务**：语义类比 (Semantics), 形态类比 (Morphology)。
  * **评价指标**：准确率 (Accuracy)。

## 核心结果摘要

### 1\. 语义相似度表现 (SimLex-999 & WordSim-353)

  * **最佳表现**：**FastText** 和 **Word2Vec (Google News)** 在相似度任务上表现最佳。
  * **架构对比**：自训练模型中，**Skip-Gram** 显著优于 CBOW，能捕捉更细粒度的语义差异。
  * **差距分析**：预训练模型由于语料规模巨大（如 Google News），在泛化能力和 OOV 处理上显著优于本地训练的 30M 语料模型。

| 模型 | SimLex-999 Spearman $\rho$ | WordSim-353 Spearman $\rho$ |
| :--- | :--- | :--- |
| **Word2Vec Google News** | **0.44** | **0.69** |
| **FastText Wiki** | **0.44** | **0.70** |
| GloVe Wiki | 0.37 | 0.61 |
| *Self-Trained Skip-Gram (Best)* | 0.26 | 0.47 |
| *Self-Trained CBOW (Best)* | 0.16 | 0.37 |

### 2\. 类比推理表现 (Google Analogy)

  * **最佳表现**：**GloVe** 在整体类比任务中准确率最高，尤其擅长语义类比（如国家-首都）。**FastText** 在形态类比（如动词时态变化）上表现最强。
  * **架构对比**：自训练模型中，**CBOW** 在类比任务上略优于 Skip-Gram，对短距离的句法特征更敏感。

| 模型 | Semantics Accuracy | Morphology Accuracy | Overall Accuracy |
| :--- | :--- | :--- | :--- |
| **GloVe Wiki** | **0.77** | 0.67 | **0.72** |
| FastText Wiki | 0.38 | **0.87** | 0.71 |
| Word2Vec Google News | 0.25 | 0.66 | 0.55 |
| *Self-Trained Models* | \< 0.10 | \< 0.15 | \< 0.13 |

## 结论

1.  **数据规模是关键**：大规模预训练模型在所有任务上均优于基于 30M 语料训练的本地模型。
2.  **模型选择建议**：
      * 若关注**语义排序**和**细粒度相似性**，推荐使用 **Skip-Gram** 或 **FastText**。
      * 若关注**全局共现统计**或**类比推理**，**GloVe** 是强有力的候选。
      * 若处理**形态丰富**或含大量**OOV**的文本，**FastText** 具有显著优势。

-----

### 依赖库 (Dependencies)

  * Python 3.x
  * Gensim \>= 4.3.2
  * spaCy \>= 3.7.5
  * NumPy
  * Pandas
  * Scipy
