# txtLoRA

纯 PyTorch 实现的文本文风 LoRA 生成与文风转换工具。

[![ModelScope Studio](https://img.shields.io/badge/ModelScope-创空间-blue)](https://modelscope.cn/studios/Vme500/txtLoRA)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 简介

txtLoRA 是一个基于 **纯 PyTorch** 实现的文本风格 LoRA（Low-Rank Adaptation）工具。它可以从少量示例文本中提取风格特征，生成 LoRA 权重，并使用该权重对任意文本进行风格转换。

- **LoRA 生成**：输入 2+ 个具有统一风格的示例文本，训练 LoRA adapter 捕获风格特征
- **文风转换**：加载训练好的 LoRA 权重，将目标文本改写为目标风格
- **一键转换**：从示例提取风格并直接转换目标文本，一步完成

## 在线体验

部署在 ModelScope 创空间，无需本地环境：

[**https://modelscope.cn/studios/Vme500/txtLoRA**](https://modelscope.cn/studios/Vme500/txtLoRA)

## 技术架构

```
txtLoRA/
├── lora.py              # 纯 PyTorch LoRA 实现（LoRALinear、权重保存/加载）
├── style_transfer.py    # 风格提取与转换（StyleLoRAModel，训练 + 生成）
├── app.py               # Gradio Web 界面（3 个标签页）
└── requirements.txt     # 依赖清单
```

### 核心技术

| 模块 | 实现 |
|------|------|
| LoRA | 纯 PyTorch，无外部 LoRA 库依赖。`LoRALinear` 包装 `nn.Linear`，添加低秩矩阵 A/B |
| 基座模型 | Qwen2.5-0.5B-Instruct，通过 ModelScope hub 自动下载 |
| 目标模块 | `q_proj`, `k_proj`, `v_proj`, `o_proj`（注意力投影层） |
| 训练方式 | 自回归语言模型（Causal LM）微调，仅更新 LoRA 参数 |
| 推理 | Chat Template 格式，精确截取 assistant 生成内容 |

### LoRA 实现细节

```python
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        # 低秩矩阵 A (in_features × rank) 和 B (rank × out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        base = self.linear(x)           # 原始输出
        lora_out = x @ self.lora_A @ self.lora_B  # LoRA 增量
        return base + lora_out * self.scaling     # 合并
```

- 训练前冻结全部模型参数，仅 LoRA A/B 矩阵可训练
- 可训练参数占比 < 0.3%，大幅降低训练成本
- 支持 merge/unmerge、权重保存/加载

## 本地运行

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- 16GB+ 内存（CPU 运行）

### 安装

```bash
git clone https://github.com/Annieif/txtLoRA.git
cd txtLoRA
pip install -r requirements.txt
```

### 启动

```bash
python app.py
```

浏览器访问 `http://localhost:7860`。

首次运行会自动从 ModelScope 下载 Qwen2.5-0.5B-Instruct 模型（约 1GB），请耐心等待。

## 使用指南

### 1. LoRA 生成

在「LoRA 生成」标签页输入示例文本（每行一个），设置 Rank 和训练轮数，点击「开始训练」。

示例：
```
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。
床前明月光，疑是地上霜。
```

训练完成后可下载 `.pt` 格式的 LoRA 权重文件。

### 2. 文风转换

在「文风转换」标签页上传 LoRA 权重，输入要转换的文本，点击「开始转换」。

### 3. 一键转换

在「一键风格提取+转换」标签页同时输入示例文本和目标文本，一步完成。

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| LoRA Rank | 低秩矩阵秩，越大捕捉风格越多 | 8 |
| 训练轮数 | 训练迭代次数 | 5-10 |
| 学习率 | 优化器学习率 | 1e-4 |
| Temperature | 生成随机性，越高越随机 | 0.8 |
| Top P | 核采样阈值 | 0.9 |

## License

MIT License - 详见 [LICENSE](LICENSE) 文件。