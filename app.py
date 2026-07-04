"""
Gradio web interface for txtLoRA - Text Style LoRA Generation & Transfer.
Pure PyTorch implementation.
"""

import gradio as gr
import torch
import os
import tempfile
import threading
import time

from style_transfer import StyleLoRAModel

# Global model instance
_model = None
_model_lock = threading.Lock()
_training_in_progress = False


def get_model():
    """Lazy-load the model (singleton)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = StyleLoRAModel(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    return _model


def generate_lora_style(
    example_texts: str,
    rank: int,
    epochs: int,
    learning_rate: float,
    progress=gr.Progress(),
):
    """
    Tab 1: Generate LoRA from example texts.
    """
    global _training_in_progress
    if _training_in_progress:
        return "训练正在进行中，请等待...", None

    texts = [t.strip() for t in example_texts.strip().split("\n") if t.strip()]
    if len(texts) < 2:
        return "请提供至少2个示例文本，每行一个", None

    _training_in_progress = True
    try:
        progress(0.0, desc="加载模型...")
        model = get_model()

        progress(0.1, desc="应用 LoRA...")
        model.apply_lora(rank=rank, alpha=rank * 2, dropout=0.05)

        progress(0.2, desc="开始训练...")
        stats = model.train_style(
            texts,
            epochs=epochs,
            batch_size=1,
            learning_rate=learning_rate,
            max_length=256,
            progress_callback=lambda epoch, batch, total_batches, loss: None,
        )

        progress(0.9, desc="保存 LoRA 权重...")
        tmp_path = os.path.join(tempfile.gettempdir(), "txtlora_style.pt")
        model.save_lora(tmp_path)

        progress(1.0, desc="完成!")

        summary = f"## 训练完成!\n\n"
        summary += f"- 示例文本数: {len(texts)}\n"
        summary += f"- LoRA Rank: {rank}\n"
        summary += f"- 训练轮数: {epochs}\n"
        summary += f"- 学习率: {learning_rate}\n\n"
        summary += "### 每轮损失:\n"
        for ep in stats["epochs"]:
            summary += f"- Epoch {ep['epoch']}: Loss = {ep['loss']:.4f}\n"
        summary += f"\n最终损失: {stats['final_loss']:.4f}"

        return summary, tmp_path
    except Exception as e:
        return f"训练失败: {str(e)}", None
    finally:
        _training_in_progress = False


def transfer_style(
    target_text: str,
    lora_file,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    progress=gr.Progress(),
):
    """
    Tab 2: Apply style transfer using trained LoRA.
    """
    if not target_text.strip():
        return "请输入要转换的文本"

    try:
        progress(0.0, desc="加载模型...")
        model = get_model()

        progress(0.2, desc="加载 LoRA 权重...")
        if lora_file is not None:
            model.load_lora(lora_file.name)
        else:
            return "请先上传 LoRA 权重文件，或在「LoRA 生成」标签页训练一个"

        progress(0.5, desc="生成中...")
        result = model.style_transfer(
            target_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        progress(1.0, desc="完成!")
        return result
    except Exception as e:
        return f"转换失败: {str(e)}"


def quick_style_transfer(
    example_texts: str,
    target_text: str,
    rank: int,
    epochs: int,
    learning_rate: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    progress=gr.Progress(),
):
    """
    Tab 3: One-click style extraction + transfer.
    """
    global _training_in_progress
    if _training_in_progress:
        return "训练正在进行中，请等待...", ""

    texts = [t.strip() for t in example_texts.strip().split("\n") if t.strip()]
    if len(texts) < 2:
        return "请提供至少2个示例文本", ""
    if not target_text.strip():
        return "请输入要转换的文本", ""

    _training_in_progress = True
    try:
        progress(0.0, desc="加载模型...")
        model = get_model()

        progress(0.05, desc="应用 LoRA...")
        model.apply_lora(rank=rank, alpha=rank * 2, dropout=0.05)

        progress(0.1, desc="从示例中提取风格...")
        stats = model.train_style(
            texts,
            epochs=epochs,
            batch_size=1,
            learning_rate=learning_rate,
            max_length=256,
        )

        progress(0.6, desc="文风转换生成中...")
        result = model.style_transfer(
            target_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        progress(1.0, desc="完成!")

        summary = f"## 训练统计\n"
        summary += f"- 示例文本数: {len(texts)}\n"
        summary += f"- Rank: {rank}, Epochs: {epochs}\n"
        for ep in stats["epochs"]:
            summary += f"- Epoch {ep['epoch']}: Loss = {ep['loss']:.4f}\n"

        return summary, result
    except Exception as e:
        return f"处理失败: {str(e)}", ""
    finally:
        _training_in_progress = False


# Build Gradio UI
def create_ui():
    css = """
    .container { max-width: 1000px; margin: auto; }
    .result-box { min-height: 150px; }
    .warning-box { background: #fff3cd; padding: 12px; border-radius: 8px; margin-bottom: 16px; }
    """

    with gr.Blocks(css=css, title="txtLoRA - 文本文风 LoRA 生成与转换") as demo:
        gr.Markdown(
            """
            # txtLoRA - 文本文风 LoRA 生成与转换
            纯 PyTorch 实现的 LoRA 文风提取与转换工具。使用 Qwen2.5-0.5B-Instruct 作为基座模型。
            """,
            elem_classes="container",
        )

        with gr.Tabs():
            # Tab 1: LoRA Generation
            with gr.TabItem("LoRA 生成"):
                gr.Markdown("### 从示例文本中提取风格，生成 LoRA 权重")
                with gr.Row():
                    with gr.Column():
                        example_input = gr.Textbox(
                            label="示例文本（每行一个）",
                            placeholder="输入具有统一风格的示例文本...\n例如：\n春眠不觉晓，处处闻啼鸟。\n夜来风雨声，花落知多少。",
                            lines=10,
                        )
                        with gr.Row():
                            rank = gr.Slider(2, 32, value=8, step=2, label="LoRA Rank")
                            epochs = gr.Slider(1, 20, value=5, step=1, label="训练轮数")
                        lr = gr.Slider(1e-5, 1e-3, value=1e-4, step=1e-5, label="学习率")
                        train_btn = gr.Button("开始训练", variant="primary", size="lg")
                    with gr.Column():
                        train_output = gr.Markdown("训练结果将显示在这里", elem_classes="result-box")
                        lora_download = gr.File(label="下载 LoRA 权重", visible=True)

                train_btn.click(
                    fn=generate_lora_style,
                    inputs=[example_input, rank, epochs, lr],
                    outputs=[train_output, lora_download],
                )

            # Tab 2: Style Transfer
            with gr.TabItem("文风转换"):
                gr.Markdown("### 使用训练好的 LoRA 权重进行文风转换")
                with gr.Row():
                    with gr.Column():
                        target_input = gr.Textbox(
                            label="要转换的文本",
                            placeholder="输入需要转换风格的文本...",
                            lines=5,
                        )
                        lora_upload = gr.File(label="上传 LoRA 权重文件 (.pt)", file_types=[".pt"])
                        with gr.Row():
                            max_tokens = gr.Slider(50, 500, value=200, step=10, label="最大生成长度")
                            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                        transfer_btn = gr.Button("开始转换", variant="primary", size="lg")
                    with gr.Column():
                        transfer_output = gr.Textbox(
                            label="转换结果",
                            lines=8,
                            interactive=False,
                        )

                transfer_btn.click(
                    fn=transfer_style,
                    inputs=[target_input, lora_upload, max_tokens, temperature, top_p],
                    outputs=[transfer_output],
                )

            # Tab 3: One-click Pipeline
            with gr.TabItem("一键风格提取+转换"):
                gr.Markdown("### 一步完成：从示例提取风格 → 直接转换目标文本")
                with gr.Row():
                    with gr.Column():
                        pipeline_examples = gr.Textbox(
                            label="示例文本（每行一个）",
                            placeholder="输入具有统一风格的示例文本...",
                            lines=8,
                        )
                        pipeline_target = gr.Textbox(
                            label="要转换的文本",
                            placeholder="输入需要转换风格的文本...",
                            lines=4,
                        )
                        with gr.Row():
                            p_rank = gr.Slider(2, 32, value=8, step=2, label="LoRA Rank")
                            p_epochs = gr.Slider(1, 20, value=5, step=1, label="训练轮数")
                        with gr.Row():
                            p_lr = gr.Slider(1e-5, 1e-3, value=1e-4, step=1e-5, label="学习率")
                            p_max_tokens = gr.Slider(50, 500, value=200, step=10, label="最大生成长度")
                        with gr.Row():
                            p_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                            p_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P")
                        pipeline_btn = gr.Button("一键转换", variant="primary", size="lg")
                    with gr.Column():
                        pipeline_stats = gr.Markdown("训练统计将显示在这里")
                        pipeline_result = gr.Textbox(
                            label="转换结果",
                            lines=8,
                            interactive=False,
                        )

                pipeline_btn.click(
                    fn=quick_style_transfer,
                    inputs=[
                        pipeline_examples, pipeline_target,
                        p_rank, p_epochs, p_lr,
                        p_max_tokens, p_temp, p_top_p,
                    ],
                    outputs=[pipeline_stats, pipeline_result],
                )

        gr.Markdown(
            """
            ---
            ### 技术说明
            - **LoRA 实现**: 纯 PyTorch，无外部依赖
            - **基座模型**: Qwen2.5-0.5B-Instruct (ModelScope)
            - **目标模块**: q_proj, k_proj, v_proj, o_proj
            - **训练方式**: 自回归语言模型 (Causal LM) 微调
            """,
            elem_classes="container",
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )