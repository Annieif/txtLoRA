"""
Text style extraction and transfer using LoRA.
Pure PyTorch implementation with HuggingFace transformers for the base model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os
import json
import random

from lora import apply_lora_to_model, get_lora_state_dict, load_lora_state_dict, count_lora_parameters


class StyleDataset(Dataset):
    """Dataset for style tuning with LoRA using instruction tuning format."""
    def __init__(
        self,
        style_texts: List[str],
        tokenizer,
        max_length: int = 512,
        augment: bool = True,
        model=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self._build_samples(style_texts, augment, model)

    def _build_samples(self, style_texts: List[str], augment: bool, model):
        """Build instruction-style training samples from style examples."""
        style_desc = self._infer_style_description(style_texts)

        for style_text in style_texts:
            # Sample 1: style_text as both input (paraphrased concept) and output
            paraphrased = self._light_paraphrase(style_text)
            self._add_sample(paraphrased, style_text, style_desc)

            # Sample 2: generic sentence -> style
            generic = self._make_generic_version(style_text)
            if generic != style_text:
                self._add_sample(generic, style_text, style_desc)

        if augment and len(style_texts) >= 2:
            # Cross-sample augmentation: mix style A concept with style B expression
            for i, style_a in enumerate(style_texts):
                for j, style_b in enumerate(style_texts):
                    if i == j:
                        continue
                    generic_a = self._make_generic_version(style_a)
                    self._add_sample(generic_a, style_b, style_desc)
                    if len(self.samples) >= 40:
                        break
                if len(self.samples) >= 40:
                    break

        # Add raw style texts for style imitation (Causal LM)
        for style_text in style_texts:
            self.samples.append({
                "full_text": style_text,
                "labels_include_all": True,
            })

        random.shuffle(self.samples)

    def _infer_style_description(self, texts: List[str]) -> str:
        """Infer a style label from sample texts."""
        text = texts[0]
        if any(p in text for p in ["。", "，", "？"]):
            if len(text) < 30 and any(c in text for c in "之乎者也矣焉哉"):
                return "古风文言"
            if any(word in text for word in ["的话", "呢", "嘛", "啦"]):
                return "口语化"
            if any(word in text for word in ["研究", "分析", "表明", "结论"]):
                return "学术论文"
        return "目标风格"

    def _light_paraphrase(self, text: str) -> str:
        """Lightweight paraphrase by swapping synonyms (rule-based)."""
        pairs = [
            ("不", "没"), ("很", "非常"), ("都", "全"),
            ("说", "讲"), ("看", "瞧"), ("好", "棒"),
            ("可以", "能够"), ("但是", "不过"), ("因为", "由于"),
        ]
        result = text
        for a, b in pairs:
            if a in result and random.random() > 0.5:
                result = result.replace(a, b, 1)
        return result if result != text else text + "（请改写）"

    def _make_generic_version(self, text: str) -> str:
        """Create a more generic/plain version of the style text."""
        result = text
        formal_words = {
            "之": "的", "矣": "了", "焉": "了", "哉": "啊",
            "乎": "吗", "乃": "是", "亦": "也", "皆": "都",
        }
        for w, r in formal_words.items():
            result = result.replace(w, r)
        return result

    def _add_sample(self, input_text: str, output_text: str, style_desc: str):
        """Add an instruction tuning sample."""
        user_msg = f"请将以下文本改写为{style_desc}，只输出改写后的文本：\n{input_text}"
        messages = [
            {"role": "system", "content": f"你是一个专业的文本风格转换助手。你的任务是将用户输入的文本改写为{style_desc}。只输出改写后的文本，不要解释，不要添加额外内容。"},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output_text},
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        self.samples.append({
            "full_text": full_text,
            "labels_include_all": False,
            "assistant_text": output_text,
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = sample["full_text"]

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        if not sample["labels_include_all"]:
            # Mask out the system+user part, only compute loss on assistant response
            # Find the last assistant turn
            assistant_text = sample.get("assistant_text", "")
            if assistant_text:
                # Tokenize just the assistant response to find its position
                # Approximate: find the position by tokenizing up to the assistant part
                # For simplicity, set all labels to -100 except the last ~len(assistant) tokens
                # Better approach: find the start of assistant in token space
                sys_user_text = full_text[:full_text.rfind(assistant_text)]
                sys_user_enc = self.tokenizer(sys_user_text, add_special_tokens=False)
                sys_user_len = min(len(sys_user_enc["input_ids"]), self.max_length)
                labels[:sys_user_len] = -100

        # Apply attention mask to labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StyleLoRAModel:
    """
    Text style LoRA model for style extraction and transfer.
    """

    # ModelScope model ID mapping
    MS_MODEL_MAP = {
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B": "qwen/Qwen2.5-0.5B",
    }

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.lora_applied = False
        self._load_model()

    def _load_model(self):
        """Load the base model and tokenizer. Try ModelScope hub first, then HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {self.model_name} on {self.device}...")

        # Try to download from ModelScope first
        model_path = self.model_name
        ms_model_id = self.MS_MODEL_MAP.get(self.model_name, None)
        if ms_model_id:
            try:
                from modelscope import snapshot_download
                print(f"Downloading from ModelScope: {ms_model_id}")
                model_path = snapshot_download(ms_model_id, cache_dir="/mnt/workspace/.cache")
                print(f"Downloaded to: {model_path}")
            except Exception as e:
                print(f"ModelScope download failed: {e}, trying HuggingFace...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def apply_lora(
        self,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        target_modules: List[str] = None,
    ):
        """Apply LoRA adapters to the model."""
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        self.lora_layers = apply_lora_to_model(
            self.model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lora_applied = True
        total, trainable = count_lora_parameters(self.model)
        print(f"LoRA applied. Total params: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    def train_style(
        self,
        texts: List[str],
        epochs: int = 10,
        batch_size: int = 2,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        rank: int = 16,
        alpha: float = 32.0,
        progress_callback=None,
    ) -> dict:
        """
        Train LoRA on example texts to capture their style.
        Uses instruction-tuning format (generic -> style) for better transfer ability.
        Returns training statistics.
        """
        if not self.lora_applied:
            self.apply_lora(rank=rank, alpha=alpha)

        dataset = StyleDataset(texts, self.tokenizer, max_length, augment=True, model=self.model)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        stats = {"epochs": [], "final_loss": 0.0}

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

                epoch_loss += loss.item()

                if progress_callback:
                    progress_callback(
                        epoch=epoch,
                        batch=batch_idx,
                        total_batches=len(dataloader),
                        loss=loss.item(),
                    )

            avg_loss = epoch_loss / len(dataloader)
            stats["epochs"].append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            scheduler.step()

        stats["final_loss"] = stats["epochs"][-1]["loss"]
        stats["num_samples"] = len(dataset)
        self.model.eval()
        return stats

    def save_lora(self, path: str):
        """Save LoRA weights to disk."""
        state = get_lora_state_dict(self.model)
        data = {
            "model_name": self.model_name,
            "state_dict": {k: {"lora_A": v["lora_A"].cpu(), "lora_B": v["lora_B"].cpu(),
                                "rank": v["rank"], "alpha": v["alpha"]}
                           for k, v in state.items()},
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(data, path)
        print(f"LoRA weights saved to {path}")
        return path

    def load_lora(self, path: str):
        """Load LoRA weights from disk."""
        if not self.lora_applied:
            self.apply_lora()

        data = torch.load(path, map_location=self.device, weights_only=False)
        for name, lora_state in data["state_dict"].items():
            parts = name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            from lora import LoRALinear
            if isinstance(module, LoRALinear):
                module.load_lora_state_dict(lora_state)
        print(f"LoRA weights loaded from {path}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        system_prompt: str = None,
    ) -> str:
        """Generate text using the base model + LoRA with chat template."""
        if system_prompt is None:
            system_prompt = "你是一个专业的文本风格转换助手。请根据用户的要求改写文本。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,
            )

        # Decode only the new tokens (skip the input prompt)
        new_tokens = outputs[0][input_len:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Fallback: if empty, decode without skipping special tokens to debug
        if not generated:
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            print(f"[DEBUG] Empty generation. Raw tokens: {raw[:200]}")
            # Try full decode
            full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant part
            marker = "assistant"
            if marker in full:
                generated = full.split(marker)[-1].strip()
            if not generated:
                generated = "(模型未生成内容，请重试或调整参数)"

        return generated

    def style_transfer(
        self,
        text: str,
        style_lora_path: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        style_description: str = "目标风格",
        style_examples: List[str] = None,
    ) -> str:
        """
        Apply style transfer to the input text.
        Uses few-shot examples in the prompt for better style alignment.
        """
        if style_lora_path:
            self.load_lora(style_lora_path)

        # Build few-shot prompt
        examples_text = ""
        if style_examples:
            examples_text = "\n\n参考风格示例：\n"
            for i, ex in enumerate(style_examples[:3]):
                examples_text += f"{i+1}. {ex}\n"

        system_prompt = (
            f"你是一个专业的文本风格转换助手。"
            f"你的任务是将用户输入的文本改写为{style_description}。"
            f"只输出改写后的文本，不要解释，不要添加任何前缀或后缀。"
            f"请确保语义不变，只改变表达风格。"
        )
        prompt = f"请将以下文本改写为{style_description}{examples_text}\n待改写文本：{text}\n改写后（直接输出，不要解释）："
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    def extract_and_transfer(
        self,
        sample_texts: List[str],
        target_text: str,
        epochs: int = 10,
        rank: int = 16,
        learning_rate: float = 2e-4,
        progress_callback=None,
    ) -> dict:
        """
        Extract style from sample texts and apply to target text.
        This is the main pipeline: train LoRA on samples, then generate with few-shot.
        """
        # Infer style description for better prompting
        from lora import apply_lora_to_model as _apply
        style_desc = "目标风格"
        if sample_texts:
            t0 = sample_texts[0]
            if any(c in t0 for c in "之乎者也矣焉哉") and len(t0) < 50:
                style_desc = "古风文言"
            elif any(w in t0 for w in ["的话", "呢", "嘛", "啦", "呀"]):
                style_desc = "口语化风格"
            elif any(w in t0 for w in ["研究", "分析", "表明", "结论", "数据"]):
                style_desc = "学术风格"
            elif any(w in t0 for w in ["!", "！", "哈哈", "卧槽"]):
                style_desc = "活泼夸张风格"

        stats = self.train_style(
            sample_texts,
            epochs=epochs,
            rank=rank,
            learning_rate=learning_rate,
            progress_callback=progress_callback,
        )

        result = self.style_transfer(
            target_text,
            style_description=style_desc,
            style_examples=sample_texts,
        )
        return {"stats": stats, "result": result}