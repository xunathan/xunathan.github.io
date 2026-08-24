---
title: vllm源码学习01:vllm整体架构，一次请求推理经过了哪些组件
date: 2026-08-23 00:00:00+0000
description: 本文是 vLLM 源码学习系列的第一篇,主要介绍一个请求进入 vLLM 之后，究竟经过了哪些组件，最终又是如何生成 Token 的。
categories:
    - 经验分享
tags:
    - vLLM
weight: 2       # You can add weight to some posts to override the default sorting (date descending)
---

## vLLM 是什么？

在开始阅读源码之前，先认识一下我们要研究的项目。

**vLLM 是一个面向大语言模型推理的高性能推理引擎。**

与直接使用 PyTorch 加载模型相比，vLLM 更关注的是如何把一个大模型变成一个能够高效处理大量用户请求的在线推理服务。

简单来说：

```text
PyTorch → 重点：把模型运行起来
vLLM    → 重点：让模型能够高效地服务大量请求
```

这意味着 vLLM 需要解决的问题，不仅仅是 Transformer 如何计算，还包括：

* 多个请求如何并发执行？
* 不同请求如何动态组成 Batch？
* GPU 计算资源如何调度？
* KV Cache 如何管理？
* 如何减少 GPU 显存浪费？
* 如何提高整体吞吐和推理效率？

因此，vLLM 可以理解成：

> **在模型之上增加了一整套面向 LLM 推理的请求调度、KV Cache 管理和模型执行机制。**

---

### 为什么选择 vLLM 学习源码？

选择 vLLM，一个很重要的原因是它刚好连接了两个层次的知识。

前面学习大模型时，我们主要关注：

```text
Token → Embedding → Attention → Transformer → Logits → Next Token
```

而学习 vLLM 后，需要进一步思考：

```text
用户请求 → 请求管理 → 调度 → Batch → KV Cache → 模型执行 → GPU → Next Token
```

前者解决的是：

> **模型是怎么工作的？**

后者解决的是：

> **怎么让模型高效地服务大量请求？**

这也是从“大模型原理”走向“大模型推理部署”的一个重要阶段。

---

## 先把 vLLM 跑起来

源码学习之前，最好先实际运行一次 vLLM。

因为如果只是直接打开源码，很容易陷入大量的类、进程、线程和数据结构。

而先把它运行起来，我们就可以建立一个简单的认识：

```text
启动 vLLM → 加载模型 → 启动服务 → 发送请求 → 得到 Token
```

后面阅读源码时，就可以不断问：

> “刚才我发送的这个请求，现在在源码中的什么位置？”

这会比单纯阅读代码更加容易建立整体认知。

---

### 安装 vLLM

vLLM 对 Python、PyTorch、CUDA、GPU 等环境有一定要求。

因此实际安装时，建议优先参考当前版本官方安装说明，而不要简单照搬网上很早的教程。

如果已经准备好了 Python 虚拟环境，可以先：

```bash
python --version
nvidia-smi
```

确认 Python 和 NVIDIA GPU 环境正常。

然后安装 vLLM：

```bash
pip install vllm
```

如果是为了**源码学习和修改代码**，则更推荐从源码安装：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

pip install -e .
```

这样后面修改 Python 源码后，可以直接重新运行，而不需要反复安装整个 vLLM。

> 注意：vLLM 对 CUDA、PyTorch、GPU 架构以及 Python 版本有较强的版本依赖。实际环境建议以当前 vLLM 官方文档和源码中的安装要求为准。

---

### 启动一个 vLLM 服务

环境准备好之后，可以使用 vLLM 提供的 OpenAI-compatible API Server。

例如：

```bash
vllm serve <model>
```

其中 `<model>` 替换成准备好的模型名称或本地模型路径。

启动后，vLLM 会完成模型加载，并启动一个 HTTP 服务。

可以把整个过程理解成：

```text
vllm serve → 创建 Engine → 初始化 EngineCore → 加载 Model → 准备 GPU → 启动 API Server → 等待 Request
```

这时候，vLLM 就从一个 Python 项目变成了一个可以接受用户请求的 LLM 推理服务。

---

### 发送一个请求

服务启动之后，可以通过 OpenAI-compatible API 发送请求。

例如：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "messages": [
      {
        "role": "user",
        "content": "介绍一下 Transformer"
      }
    ],
    "max_tokens": 100
  }'
```

如果服务正常运行，就会得到类似这样的结果：

```text
{
    "choices": [
        {
            "message": {
                "content": "Transformer 是一种基于注意力机制的..."
            }
        }
    ]
}
```

到这里，我们已经完成了一次最简单的 vLLM 推理：

```text
Client → HTTP Request → vLLM API Server → Engine → EngineCore → Scheduler → Model → GPU → Generated Tokens → HTTP Response
```

这张图也正是我们接下来要进入源码研究的主线。

---

### 从运行结果回到源码

现在再回头看这个过程，会发现一个很有意思的问题。

我们只是执行了：

```bash
vllm serve ...
```

然后发送了一次 HTTP 请求。

但 vLLM 内部实际上发生了很多事情：

```text
HTTP Request → 创建 Request → Engine 接收 → EngineCore 处理 → Scheduler 调度
  → Prefill → Decode → KV Cache → Sampling → Next Token → 返回结果
```

**接下来的源码学习，就是把这条看不见的链路一点一点找出来。**

所以这篇文章后面不会按照源码目录逐个介绍类，而是采用一种更容易理解的方式：

> **跟着刚才发送的那一个请求，看看它到底是怎么穿过 vLLM 源码的。**

---

## 背景回顾：一次 Token 是怎么生成的？

在进入 vLLM 源码之前，先简单回顾一下大模型生成 Token 的过程。

假设用户输入：

```text
介绍一下 Transformer
```

首先经过 Tokenizer：

```text
文本
 ↓
Tokenizer
 ↓
[Token1, Token2, Token3, ...]
```

然后这些 Token 会经过 Embedding 和 Transformer 网络：

```text
Tokens
   ↓
Embedding
   ↓
Transformer
   ↓
Logits
   ↓
Sampling
   ↓
Next Token
```

例如模型生成：

```text
介绍一下 Transformer
        ↓
Transformer 是一种
        ↓
Transformer 是一种基于
        ↓
Transformer 是一种基于注意力
        ↓
...
```

这个过程可以简单理解成：

> **模型根据已有 Token，预测下一个 Token，然后把新 Token 加回来，继续预测。**

因此，一次完整生成实际上是一个循环：

```text
Prompt
  ↓
Prefill
  ↓
生成 Token
  ↓
Decode
  ↓
生成 Token
  ↓
Decode
  ↓
...
  ↓
结束
```

但在真实服务中，往往不是只有一个请求。

```text
Request A ──→ Decode
Request B ──→ Prefill
Request C ──→ 等待调度
Request D ──→ Decode
```

这时，**什么时候执行哪个请求、执行多少 Token、如何管理 KV Cache**，就成为推理框架需要解决的问题。

接下来，我们看看 vLLM 是怎么组织这些工作的。

---

## vLLM 架构全景

在深入源码之前，先建立一个整体的地图，不然很容易一头扎进某个类里出不来，忘了自己是从哪条路走进去的。

回到刚才那次请求。一个 HTTP 请求从 curl 发出去，到最后拿到生成的 Token，中间大致经过了这样几层：

```text
Client
  → API Server（接收 HTTP 请求，解析成内部数据结构）
  → Engine（面向上层的统一入口）
  → EngineCore（真正驱动推理循环的核心）
  → Scheduler（决定这一步该跑哪些请求、怎么组 Batch）
  → Worker / Model Runner（在 GPU 上真正执行模型前向计算）
  → Model / Attention（Transformer 本身，包括 PagedAttention）
```

不同版本的 vLLM 在具体类名和模块划分上可能有所变化，但从整体职责来看，可以先这样理解。

这几层的关系，可以用一句话概括：

API Server 负责"对外"，Engine 和 EngineCore 负责"调度"，Worker 和 Model 负责"计算"。

它们不是简单的调用链，而是有分工的：

* API Server 只关心协议层的事情——怎么解析 OpenAI 格式的请求、怎么把生成结果流式返回给客户端。它不关心 Attention 怎么算，也不关心 KV Cache 放在哪块显存。
* Engine / EngineCore 是整个系统的"大脑"，负责把源源不断进来的请求管理起来，决定谁先执行、以什么方式组成 Batch，什么时候该做 Prefill、什么时候该做 Decode。这一层完全不涉及具体的模型计算，它面对的是抽象的"请求"和"资源"。
* Scheduler 是 EngineCore 里最关键的一个组件，专门负责"排课"：GPU 显存和计算资源是有限的，Scheduler 要在有限资源下决定这一轮该处理哪些请求，这也是 Continuous Batching 真正发生的地方。
* Worker / Model Runner 是真正"干活"的一层，拿到 Scheduler 排好的 Batch 之后，负责把数据搬到 GPU 上、调用模型做前向计算、把结果吐出来。
* Model / Attention 就是我们熟悉的 Transformer 结构了，只不过 vLLM 在这里加了一层 PagedAttention，让 KV Cache 可以像操作系统管理内存一样被灵活地分页管理。

值得一提的是，**KV Cache 的管理并不是单独的一层**，而是贯穿在 Scheduler 和 Worker 之间——Scheduler 决定"这个请求该用多少块 KV Cache、放在哪里"，Worker 在真正计算时去读写这些 Cache。这也是为什么后面走源码的时候，KV Cache 会在好几个地方反复出现，而不是只在某一个文件里。

有了这张地图，接下来我们就不再"猜"某个组件是干什么的了，而是带着这张图，去看一个真实的请求是怎么依次穿过这几层的。

---

## 跟着一次请求走一遍 vLLM 源码

假设现在用户发送一个请求：

```text
“介绍一下 Transformer”
```

我们就跟着这个请求，看它在 vLLM 中经历了什么。

---

### 请求进入与创建

请求首先从 API Server 进入 vLLM。

可以简单理解为：

```text
HTTP Request
     ↓
API Server
     ↓
Engine
     ↓
创建 Request
```

这个 Request 并不是简单保存一段字符串。

它还需要记录后续推理过程中需要的信息，例如：

```text
Request
 ├── request_id
 ├── prompt
 ├── tokens
 ├── sampling parameters
 └── request status
```

例如用户可能指定：

```text
max_tokens = 100
temperature = 0.7
```

这些信息都会随着 Request 一起进入后续的推理流程。

到这里，一个外部请求就变成了 vLLM 内部可以管理的 Request。

接下来最关键的问题来了：

> **现在有很多 Request，GPU 下一步到底应该执行谁？**

这就是 Scheduler 的工作。

---

### Scheduler 调度

Scheduler 可以理解成 vLLM 中的“交通指挥员”。

假设当前有：

```text
Request A：正在 Decode
Request B：刚刚进入
Request C：正在 Decode
Request D：等待执行
```

Scheduler 每一轮都会根据当前的资源和请求状态，决定哪些请求可以执行。

简单表示：

```text
              Scheduler
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Request A   Request B   Request C
```

这里就涉及 vLLM 一个非常重要的思想：

> **Continuous Batching（连续批处理）**

传统 Batch 往往是：

```text
Batch
 ├── A
 ├── B
 ├── C
 └── D

一起开始
一起结束
```

而推理服务中的请求并不会这么整齐。

可能：

```text
A ──────────────── 完成
B ─────────── 完成
C ───────────────────────
       D ────────────
```

因此 vLLM 会根据每一轮的请求状态动态调整 Batch。

例如：

```text
第 1 轮：
[A, B, C]

第 2 轮：
[A, C, D]

第 3 轮：
[C, D, E]
```

请求可以不断进入，已经完成的请求也可以退出。

这就是 Continuous Batching 的核心思想。

在源码中，你会看到 Scheduler 不断处理：

```text
Request
    ↓
状态判断
    ↓
资源判断
    ↓
选择本轮执行的 Request
```

最终形成这一轮真正需要执行的请求集合。

然后这些请求会交给 Worker 执行。

---

### 4.3 Prefill 执行

如果这是一个刚刚进入系统的新请求，它首先需要进行 Prefill。

例如：

```text
“介绍一下 Transformer”
```

经过 Tokenizer 后得到：

```text
[Token1, Token2, Token3, Token4]
```

Prefill 会一次性处理这些输入 Token：

```text
Token1 Token2 Token3 Token4
          ↓
      Transformer
          ↓
       Attention
          ↓
       KV Cache
          ↓
        Logits
```

Prefill 的主要任务可以简单理解成：

> **让模型先“读完”用户输入，并建立后续生成需要的状态。**

在源码中，我们可以看到 Scheduler 最终产生的请求会进入模型执行流程，Model Runner 准备输入数据，然后调用模型进行前向计算。

整体过程可以简化为：

```text
Scheduler
    ↓
Model Runner
    ↓
Model
    ↓
Transformer
    ↓
Attention
    ↓
Logits
```

Prefill 完成后，模型就可以开始生成第一个新的 Token。

---

### 4.4 Decode 执行

Prefill 完成之后，就进入 Decode 阶段。

假设模型生成：

```text
“Transformer”
```

下一轮需要根据已经生成的内容继续预测：

```text
已有 Tokens
     +
新生成 Token
     ↓
Decode
     ↓
Next Token
```

例如：

```text
第 1 次 Decode → “Transformer”
第 2 次 Decode → “是一种”
第 3 次 Decode → “神经网络”
第 4 次 Decode → “架构”
...
```

这个过程会不断重复：

```text
Decode
  ↓
Next Token
  ↓
Decode
  ↓
Next Token
  ↓
Decode
  ↓
...
```

与 Prefill 不同，Decode 阶段通常每次只需要处理新生成的 Token。

而之前已经计算过的历史信息，会通过 KV Cache 复用。

因此：

```text
Prefill
  → 处理整个 Prompt

Decode
  → 每次处理新 Token
  → 复用历史 KV Cache
```

这也是理解 LLM 推理性能的一个关键。

---

### 4.5 KV Cache 的读写

在 Transformer 的 Attention 中，每次计算都会产生 K 和 V。

如果每生成一个 Token，都重新计算之前所有 Token 的 K/V，计算量会非常大。

因此，之前计算得到的 K/V 会被保存下来：

```text
Token
 ↓
Attention
 ↓
K / V
 ↓
KV Cache
```

后续 Decode 时：

```text
新 Token
   +
历史 KV Cache
   ↓
Attention
   ↓
新的 K / V
   ↓
更新 KV Cache
```

可以简单理解成：

```text
第一次：
Prompt → 计算 KV → 保存

第二次：
新 Token + 历史 KV → 计算 → 更新 KV

第三次：
新 Token + 历史 KV → 计算 → 更新 KV
```

所以 KV Cache 实际上贯穿了整个 Decode 过程。

而这也带来了一个现实问题：

> **KV Cache 会占用大量 GPU 显存。**

当同时运行大量请求时，如何高效管理这些 KV Cache，就变成了 vLLM 的核心问题之一。

在后续源码学习中，我们会进一步看到 vLLM 如何通过 KV Cache Manager、Block 等机制管理这些数据。

这一篇先把它理解成：

> **KV Cache 保存了请求历史 Token 的 Attention 状态，让 Decode 可以复用之前的计算结果。**

---

### 4.6 Sampling 与请求结束

模型经过前向计算后，会得到每个 Token 的概率，也就是 Logits。

```text
Model
 ↓
Logits
 ↓
Sampling
 ↓
Next Token
```

例如：

```text
“Transformer” → 0.35
“是一种”     → 0.21
“模型”       → 0.08
...
```

Sampling 根据用户设置的参数选择最终 Token。

例如：

```text
temperature
top_k
top_p
```

然后得到：

```text
Next Token
```

这个 Token 会加入当前 Request。

如果还没有结束：

```text
Request
   ↓
继续进入 Scheduler
   ↓
下一轮 Decode
```

于是整个过程形成一个循环：

```text
             ┌───────────────┐
             │   Scheduler   │
             └───────┬───────┘
                     ↓
               Model Runner
                     ↓
                   Model
                     ↓
                  Logits
                     ↓
                 Sampling
                     ↓
                 Next Token
                     │
            ┌────────┴────────┐
            │                 │
          未结束              已结束
            │                 │
            ↓                 ↓
       下一轮 Decode         返回结果
            │
            └────→ Scheduler
```

直到满足结束条件，例如：

* 生成 EOS Token
* 达到 `max_tokens`
* 请求被取消

最终生成结果返回给用户。

---

## 从这一次请求看 vLLM 的设计思想

跟着一个请求走下来，可以发现：

```text
Request
   ↓
Engine
   ↓
EngineCore
   ↓
Scheduler
   ↓
Worker
   ↓
Model
   ↓
Sampling
   ↓
Next Token
```

表面上看，这是一次普通的模型推理。

但 vLLM 真正做的事情，是在这条模型推理链路之外，增加了一整套**请求管理、调度和 GPU 资源管理机制**。

可以把它总结成三个核心思想。

### 1. 模型执行只是其中一部分

vLLM 不只是：

```text
Input → Model → Output
```

而是：

```text
Request Management
        ↓
Scheduling
        ↓
Resource Management
        ↓
Model Execution
        ↓
Output
```

---

### 2. Scheduler 是连接“请求”和“GPU”的关键

用户的请求是动态的：

```text
不断进入
不断完成
```

而 GPU 是有限的。

Scheduler 就负责在两者之间进行协调：

```text
大量 Requests
      ↓
   Scheduler
      ↓
有限 GPU 计算资源
```

---

### 3. KV Cache 是整个推理过程的重要资源

Decode 阶段不断依赖历史 KV：

```text
历史 Token
    ↓
KV Cache
    ↓
Decode
    ↓
新的 KV
    ↓
KV Cache
```

因此，如何高效地管理 KV Cache，也是 vLLM 性能设计的重要组成部分。

---

## 写到这里，我们建立了 vLLM 的第一张地图

现在再回头看 vLLM，就不再只是一个个陌生的类和目录：

```text
API Server
    ↓
Engine
    ↓
EngineCore
    ↓
Scheduler
    ↓
Worker
    ↓
Model
    ↓
GPU
```

而是一条完整的推理链路：

```text
用户请求
   ↓
创建 Request
   ↓
Scheduler 调度
   ↓
Prefill
   ↓
KV Cache
   ↓
Decode
   ↓
Sampling
   ↓
Next Token
   ↓
继续 Decode
   ↓
请求结束
```

这也是后续源码学习的主线。

下一篇将不再停留在整体架构，而是从 **EngineCore** 开始，深入看看：

> **EngineCore 是如何启动的？为什么它会运行在独立进程中？主进程和 EngineCore 又是如何建立通信并完成握手的？**

从这里开始，正式进入 vLLM 源码。
