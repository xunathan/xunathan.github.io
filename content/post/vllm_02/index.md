---
title: vllm源码学习02:主进程如何创建EngineCore及进程间如何通讯？
date: 2026-08-23 00:00:00+0000
description: vllm推理框架
categories:
    - 经验分享
tags:
    - vLLM
weight: 2       # You can add weight to some posts to override the default sorting (date descending)
---

① EngineCore 为什么是独立进程？

② 主进程和 EngineCore 怎么建立通信？

③ 为什么创建进程之后还需要一次 Handshake？

## 为什么 vLLM 需要 EngineCore 子进程？


## 主进程如何创建并启动EngineCore 子进程


## 主进程和EngineCore 子进程的通讯：handshake

## 推理请求的发送

## 总结