---
title: 一句话是如何进入大模型的？
date: 2026-08-02 00:00:00+0000
description: 本文为跟着大模型的数据流，看懂一个大模型到底在干什么系列的第一篇，讲述了一句话是以什么形式进入大模型的。
categories:
    - 经验分享
tags:
    - LLM
weight: 2       # You can add weight to some posts to override the default sorting (date descending)
---

## 大模型
大模型现在已经融入大家工作学习的很多方面了，比如对话问答、写代码、做视频等。有没有想过，它到底是怎么工作的？
答案简单的令所有人吃惊，**猜下一个词。**不是思考得出来的，也不是真的理解了这段话，就是给定了前面的文字，直接预测下一个
最可能出现的词。市面上的大模型，chatgpt、deepseek、千问等基本原理都是：**预测下一个词。**

### 运行环境说明
本系列的blog会用实际代码一步步演示大模型从输入到输出的矩阵变换，需要跑简单的大模型代码。采用python语言演示代码示例，需要的python库为**transformers、torch**，
打算以gpt2大模型一步一步剖析大模型的内部结构，不需要GPU环境。

### 为什么计算机不能理解文字

人类阅读文字时，看到的是汉字、单词、句子，并能够直接理解其中的含义；但对于计算机来说，它只能处理数字，无法直接理解自然语言。因此，大模型在真正开始计算之前，必须先将文本转换成一种既能保留语言信息，又便于计算机处理的数字表示，而 Token 正是连接自然语言和神经网络之间的桥梁。

如果直接以字符或单词作为处理单位，会遇到很多问题。以字符为单位，虽然词表较小，但一句话会被拆分得非常零碎，模型需要处理更长的序列，难以学习完整的语义；以单词为单位，则需要维护一个庞大的词表，不仅存储成本高，还无法很好地处理新词、专业术语、人名以及不同语言。随着互联网每天不断产生新的表达方式，单纯依赖固定的单词词典几乎是不现实的。

因此，大模型普遍采用一种折中的方案——Token。Token 并不等同于一个汉字或一个单词，它可以是一个完整的单词、一个词根、一个标点符号，甚至是一个汉字或几个汉字组成的短语。通过专门的分词算法（如 BPE、WordPiece、SentencePiece），模型能够将任意文本拆分成有限数量的 Token，并建立一张固定大小的 Token 词表。这样，无论输入的是中文、英文还是代码，最终都能够表示成一串 Token 序列。

例如，句子 "ChatGPT is amazing!" 可能会被拆分为 ["Chat", "GPT", " is", " amazing", "!"]；而中文 "今天天气很好" 在不同模型中，可能会被拆分为 ["今天", "天气", "很好"]，也可能拆分为 ["今", "天", "天气", "很", "好"]。不同模型采用的 Tokenizer 不同，因此同一句话得到的 Token 数量也可能不同。

正因为 Token 兼顾了表达能力和计算效率，它已经成为现代大模型事实上的"语言单位"。后续的词嵌入（Embedding）、位置编码（Positional Encoding）、注意力机制（Attention）以及 Transformer 的所有计算，都是围绕 Token 展开的。可以说，理解了 Token，就理解了大模型处理自然语言的第一步。

### Tokenizer做了什么

理解了 Token 的作用之后，一个新的问题随之而来：**模型是如何把一段普通的文本拆分成 Token 的？**

答案就是 **Tokenizer（分词器）**。

Tokenizer 可以理解为大模型的"语言翻译官"。它位于用户输入和神经网络之间，负责将人类能够理解的自然语言，转换成模型能够处理的 Token 序列。整个过程发生在模型真正开始推理之前，因此无论是 ChatGPT、DeepSeek、Llama，还是其他大语言模型，都会首先经过 Tokenizer 的处理。

从整体来看，Tokenizer 的工作可以分为四个步骤。

#### 第一步：文本规范化（Normalization）

Tokenizer 首先会对输入文本进行预处理，例如统一不同的编码格式、处理换行符、规范空格、统一大小写（部分模型会这样做）等。

例如：

```
Hello   World!
```

经过规范化之后可能变成：

```
Hello World!
```

不同模型采用的规范化策略不同，但目的都是减少无意义的文本差异，使相同含义的文本尽可能得到一致的处理结果。

---

#### 第二步：文本切分（Pre-tokenization）

完成规范化之后，Tokenizer 会先按照一定规则进行初步切分。

例如下面这句话：

```
ChatGPT is amazing!
```

首先可能被切分为：

```
["ChatGPT", "is", "amazing", "!"]
```

对于中文：

```
今天天气很好
```

由于中文本身没有空格，不同 Tokenizer 会采用不同策略，有的会先按字符切分：

```
["今", "天", "天", "气", "很", "好"]
```

也有一些 Tokenizer 会直接结合训练得到的词表进行匹配，而不会严格经历这一过程。

这一步只是初步划分文本边界，并不是最终的 Token。

---

#### 第三步：子词编码（Subword Tokenization）

这是整个 Tokenizer 最核心的一步。

前面介绍过，如果直接按照完整单词建立词典，会导致词表无限增大；如果按单个字符切分，又会丢失很多语义信息。因此，大模型普遍采用 **子词（Subword）** 作为基本单位。

Tokenizer 会利用训练阶段已经生成好的词表，对文本不断寻找能够匹配的最长 Token。

例如：

```
unbelievable
```

可能不会作为一个完整 Token，而是拆分为：

```
["un", "believ", "able"]
```

对于英文中的新单词：

```
ChatGPT
```

可能拆成：

```
["Chat", "GPT"]
```

对于中文：

```
机器学习
```

可能直接成为：

```
["机器学习"]
```

也可能拆成：

```
["机器", "学习"]
```

甚至：

```
["机", "器", "学习"]
```

最终如何拆分，完全取决于模型训练时生成的词表（Vocabulary）。

目前主流模型大多采用 BPE、WordPiece 或 SentencePiece 等算法来完成这一步。

---

#### 第四步：映射为 Token ID

完成 Token 切分之后，模型仍然不能直接进行计算。

因为 Transformer 并不能识别字符串，它只能处理数字。

因此，每一个 Token 都会在词表（Vocabulary）中找到唯一对应的编号（Token ID）。

例如，一个模型的词表中可能存在如下映射：

| Token   | Token ID |
| ------- | -------- |
| Chat    | 15231    |
| GPT     | 9812     |
| is      | 318      |
| amazing | 4998     |
| !       | 0        |

于是：

```
ChatGPT is amazing!
```

最终就变成：

```
[15231, 9812, 318, 4998, 0]
```

这就是模型真正接收到的输入。

后面的 Embedding 层会根据这些 Token ID 查找对应的向量表示，再送入 Transformer 进行计算。

---

#### 整个 Tokenizer 流程

综合来看，一段文本从输入到模型，中间会经历如下流程：

```
原始文本
      │
      ▼
文本规范化（Normalization）
      │
      ▼
初步切分（Pre-tokenization）
      │
      ▼
子词编码（BPE / WordPiece / SentencePiece）
      │
      ▼
Token 序列
      │
      ▼
Token ID 序列
      │
      ▼
Embedding
      │
      ▼
Transformer
```

可以看到，Tokenizer 本身并不会理解文本的含义，它只是按照预先训练好的规则，把文本转换成模型统一使用的 Token，并进一步映射为数字编号。真正对语言进行理解和推理的是后续的 Embedding 层和 Transformer 网络。

因此，可以把 Tokenizer 看成整个大模型的数据入口，它决定了文本如何被拆分，也决定了模型最终看到的输入形式。

#### 代码示例
代码示例我们加载大模型一个大模型，我们使用transformers自带的gpt2模型，和当前最新的gpt以及主流大模型就是规模不同，架构都是相同的。如果你的网络访问不了，可以使用qwen3:0.6B这个模型，模型可以本地下载。这两个都是CPU都可以跑的模型。




```python
# pip install transformers torch
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#from transformers import AutoTokenizer, AutoModelForCausalLM，本地加载qwen3使用这两个库
import torch

# 加载模型和分词器
print("正在加载模型（首次运行会下载约 500MB）…")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2") #分词器
model = GPT2LMHeadModel.from_pretrained("gpt2") #模型
model.eval()

#输入一句话
prompt = "Hello World!"
print(f"\nInput: {prompt}")
#文字转为模型能懂的Token Id
input_ids = tokenizer.encode(prompt, return_tensors="pt")
tokens = [tokenizer.decode(id) for id in input_ids[0]]
print(f"Token IDs: {input_ids.tolist()[0]}")
print(f"对应的 tokens: {tokens}")
print(f"Token 数量: {len(tokens)}")

```

    正在加载模型（首次运行会下载约 500MB）…



    Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]


    
    Input: Hello World!
    Token IDs: [15496, 2159, 0]
    对应的 tokens: ['Hello', ' World', '!']
    Token 数量: 3


### Embedding
## Embedding：把 Token ID 转换成模型能够理解的向量

经过 Tokenizer 的处理后，一段文本已经变成了一串 Token ID，例如：

```text
"ChatGPT is amazing!"
        │
        ▼
[15231, 9812, 318, 4998, 0]
```

看到这里，很多人会产生一个疑问：

> **既然已经转换成数字了，为什么还需要 Embedding？**

原因很简单：**Token ID 只是一个编号，而编号本身并没有任何语义。**

例如，在某个模型的词表中：

| Token | Token ID |
| ----- | -------: |
| cat   |       25 |
| dog   |       26 |
| apple |       27 |

如果直接把这些数字送入神经网络，模型很可能会错误地认为：

* dog（26）比 cat（25）"大一点"
* apple（27）比 dog（26）更接近

但实际上，这些数字只是词表中的索引，**25、26、27 之间不存在任何数学意义，也无法反映词语之间的语义关系。**

因此，大模型需要一种新的表示方式，让具有相似含义的词，在数学空间中也更加接近。

这就是 **Embedding（词嵌入）** 的作用。

### Embedding 的本质

Embedding 可以理解为一张巨大的"词典"，只不过词典中存储的不再是文字解释，而是一组数字组成的向量。

例如：

| Token | Token ID | Embedding（示意）               |
| ----- | -------: | --------------------------- |
| cat   |       25 | `[0.18, -0.62, 0.91, ...]`  |
| dog   |       26 | `[0.21, -0.59, 0.87, ...]`  |
| apple |       27 | `[-0.73, 0.45, -0.11, ...]` |

可以发现：

* **cat** 和 **dog** 的向量非常相似；
* **apple** 的向量则相差较大。

这意味着，在高维向量空间中，模型能够通过向量之间的距离来表达词语之间的语义关系，而不是依赖毫无意义的编号。

因此，Transformer 真正处理的并不是 Token ID，而是这些向量。

---

### Embedding 矩阵

从程序实现来看，Embedding 本质上就是一个二维矩阵。

假设模型拥有：

* Vocabulary（词表）大小：50,000
* Hidden Size（隐藏维度）：768

那么 Embedding 矩阵就是：

```text
Embedding Matrix
┌──────────────────────────────┐
│ Token0   → 768 个数字         │
│ Token1   → 768 个数字         │
│ Token2   → 768 个数字         │
│ ...                          │
│ Token49999 → 768 个数字       │
└──────────────────────────────┘
```

整个矩阵可以表示为：

```text
50000 × 768
```

矩阵中的每一行都对应一个 Token 的向量。

例如：

```text
Token ID = 318
```

模型实际上执行的是：

```python
embedding = embedding_table[318]
```

也就是说，**Embedding 并不是复杂的计算，而是一次查表（Lookup）操作。**

---

### 一个简单的例子

假设词表中只有四个 Token：

| Token | ID |
| ----- | -: |
| 我     |  0 |
| 爱     |  1 |
| AI    |  2 |
| 。     |  3 |

Embedding 矩阵如下（为了演示，仅使用 4 维向量）：

```text
ID    Embedding
0  → [0.2, 0.5, -0.1, 0.8]
1  → [0.7, 0.1,  0.4, 0.3]
2  → [0.9, 0.8, -0.2, 0.5]
3  → [0.1, 0.2,  0.1, 0.4]
```

输入一句话：

```text
我 爱 AI 。
```

Tokenizer 输出：

```text
[0, 1, 2, 3]
```

Embedding 查表之后：

```text
[
 [0.2, 0.5, -0.1, 0.8],
 [0.7, 0.1,  0.4, 0.3],
 [0.9, 0.8, -0.2, 0.5],
 [0.1, 0.2,  0.1, 0.4]
]
```

此时，原来的文本已经变成了一个由多个向量组成的矩阵，这才是 Transformer 真正接收到的输入。

---

### Embedding 是如何得到的？

很多初学者会误以为，这些向量是人工设计出来的。

事实上并不是。

在模型刚开始训练时，Embedding 矩阵通常是随机初始化的，每个 Token 都对应一组随机数字。随着模型不断学习海量文本，训练算法会根据预测误差不断调整这些向量。经过数十亿甚至数万亿 Token 的训练后，语义相近的词会逐渐聚集到向量空间中的相近位置，而语义不同的词则会彼此远离。

因此，Embedding 并不是人为赋予语义，而是模型在训练过程中"学习"出来的知识表示。

---

### 小结

Embedding 的作用可以概括为一句话：

> **Embedding 将没有语义的 Token ID，转换成包含语义信息的高维向量，让 Transformer 能够对语言进行数学计算。**

整个数据流到这里已经变成：

```text
原始文本
    │
Tokenizer
    │
Token
    │
Token ID
    │
Embedding（查表）
    │
向量（Vector）
    │
Transformer
```

从下一节开始，我们将介绍另一个同样重要的问题：**Transformer 为什么还需要位置编码（Positional Encoding）？** 毕竟，经过 Embedding 后，每个 Token 都只是一个独立的向量，模型如何知道"谁在前、谁在后"呢？答案就在位置编码中。
### 代码示例



```python
embedding_table = model.transformer.wte.weight.detach()
print(f"\nEmbedding 表的形状: {embedding_table.shape}")
print(f"  行数（词表大小）: {embedding_table.shape[0]}")
print(f"  列数（向量维度）: {embedding_table.shape[1]}")
print(f"  总参数量: {embedding_table.shape[0] * embedding_table.shape[1]:,}")
```

    
    Embedding 表的形状: torch.Size([50257, 768])
      行数（词表大小）: 50257
      列数（向量维度）: 768
      总参数量: 38,597,376



## 位置编码（Positional Encoding）：让模型知道 Token 的先后顺序

经过 Embedding 之后，每一个 Token 都已经被转换成了一个高维向量。例如，句子：

```text
我 爱 AI
```

经过 Tokenizer 和 Embedding 后，可能表示为：

```text
[
  [0.2, 0.5, ...],   ← 我
  [0.7, 0.1, ...],   ← 爱
  [0.9, 0.8, ...]    ← AI
]
```

这些向量已经包含了一定的语义信息，但它们还有一个致命的问题：

> **模型不知道这些 Token 的前后顺序。**

换句话说，对于 Transformer 来说，它看到的只是三个向量，而不是一句有顺序的语言。

例如下面两句话：

```text
我 爱 你
```

和

```text
你 爱 我
```

虽然使用的是完全相同的三个 Token，但它们表达的含义完全不同。

然而，如果只依赖 Embedding，这两个句子只是同一组向量的不同排列。由于 Transformer 的注意力机制本身并不会自动理解"第一个""第二个""第三个"这样的位置信息，因此模型无法仅凭 Embedding 判断两个句子的区别。

因此，我们需要额外告诉模型：

* 哪个 Token 在第一个位置；
* 哪个 Token 在第二个位置；
* 哪个 Token 在第三个位置……

这就是**位置编码（Positional Encoding）**存在的意义。

---

### 最简单的理解：给每个 Token 加一个"位置标签"

可以把位置编码理解成给每个 Token 增加一个隐藏的"位置标签"。

例如：

```text
我   爱   AI
```

除了语义向量之外，每个 Token 还拥有自己的位置编号：

| Token | 位置（Position） |
| ----- | -----------: |
| 我     |            0 |
| 爱     |            1 |
| AI    |            2 |

模型随后会根据位置编号，为每个位置生成一个对应的位置向量（Position Vector）。

例如（示意）：

```text
Position 0 → [0.10, 0.08, ...]
Position 1 → [0.32, 0.17, ...]
Position 2 → [0.51, 0.42, ...]
```

最后，把语义向量和位置向量逐元素相加：

```text
最终输入 = Embedding + Position Embedding
```

例如：

```text
Embedding：

我  → [0.20, 0.50]
爱  → [0.70, 0.10]
AI → [0.90, 0.80]

Position：

P0 → [0.10, 0.20]
P1 → [0.30, 0.10]
P2 → [0.40, 0.50]
```

相加之后：

```text
我  → [0.30, 0.70]
爱  → [1.00, 0.20]
AI → [1.30, 1.30]
```

这样，每个 Token 不仅包含了**"它是什么"（语义）**，还包含了**"它在哪里"（位置）**。

Transformer 后续处理的，就是这些融合了语义和位置信息的向量。

---

### GPT 为什么采用 Position Embedding？

最初的 Transformer（论文《Attention Is All You Need》）采用的是**正弦/余弦位置编码（Sinusoidal Positional Encoding）**。

这种方法利用不同频率的正弦函数计算位置向量，不需要学习参数，因此能够自然支持比训练时更长的序列。

不过，GPT 系列模型没有采用这种固定公式，而是使用了**可学习的位置编码（Learned Positional Embedding）**。

也就是说，模型维护了一张位置编码表：

```text
Position Embedding Table

Position 0 → 向量
Position 1 → 向量
Position 2 → 向量
...
Position 1023 → 向量
```

每个位置对应的向量都会随着模型训练不断更新，就像 Embedding 矩阵一样，由数据自动学习得到。

对于 GPT-2 来说，模型支持最长 **1024** 个 Token，因此位置编码表共有 **1024 行**；如果隐藏维度为 **768**，那么位置编码矩阵的大小就是：

```text
1024 × 768
```

在推理时，第 0 个 Token 查找第 0 行，第 1 个 Token 查找第 1 行，以此类推，然后与对应的 Token Embedding 相加。

---

### GPT-2 中的数据流

到目前为止，一句话已经经历了如下处理流程：

```text
原始文本
    │
Tokenizer
    │
Token
    │
Token ID
    │
Embedding Lookup
    │
Token Embedding
    │
Position Embedding Lookup
    │
Token Embedding + Position Embedding
    │
Transformer Block
```

可以看到，在真正进入 Transformer 之前，每个 Token 已经同时拥有了**语义信息**和**位置信息**。

---
### Qwen的位置编码
GPT2的位置编码可以简单理解为，比如上下文长度为1024，也即是有1024个位置，位置编码的向量必须和embedding向量的维度相同，也就是必须为768，我们可以类比计算机的数字表示，比如用4个字节32Bit表示一个INT数字。位置编码这里可以理解为用768个bit编码出一个数字，这个数字就表示位置。位置编码要加入到Embedding向量一起进入大模型。
位置编码有一定的局限性，当前流行的为RoPE（旋转位置编码），有兴趣的可自行学习。

### 小结

位置编码的作用可以概括为一句话：

> **Embedding 告诉模型"这个 Token 是什么"，位置编码告诉模型"这个 Token 在哪里"。**

两者缺一不可。如果只有 Embedding，模型能够理解每个 Token 的语义，却无法区分它们的排列顺序；加入位置编码之后，Transformer 才能够理解"我爱你"和"你爱我"这样的句子为什么含义不同。

完成位置编码后，输入向量就已经准备就绪。下一步，这些向量将进入 Transformer 中最核心的模块——**Self-Attention（自注意力机制）**。在那里，每个 Token 会与序列中的其他 Token 建立联系，从上下文中动态获取信息，这也是大模型能够理解长文本和复杂语义的关键所在。



