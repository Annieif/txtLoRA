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

from lora import apply_lora_to_model, get_lora_state_dict, load_lora_state_dict, count_lora_parameters


class StyleDataset(Dataset):
    """Dataset for style tuning with LoRA."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }


class StyleLoRAModel:
    """
    Text style LoRA model for style extraction and transfer.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.lora_applied = False
        self._load_model()

    def _load_model(self):
        """Load the base model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=None,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def apply_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: List[str] = None,
    ):
        """Apply LoRA adapters to the model."""
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

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
        epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        max_length: int = 256,
        progress_callback=None,
    ) -> dict:
        """
        Train LoRA on example texts to capture their style.
        Returns training statistics.
        """
        if not self.lora_applied:
            self.apply_lora()

        dataset = StyleDataset(texts, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

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

        stats["final_loss"] = stats["epochs"][-1]["loss"]
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
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text using the base model + LoRA."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):] if generated.startswith(prompt) else generated

    def style_transfer(
        self,
        text: str,
        style_lora_path: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
    ) -> str:
        """
        Apply style transfer to the input text.
        If style_lora_path is provided, load that LoRA; otherwise use current LoRA.
        """
        if style_lora_path:
            self.load_lora(style_lora_path)

        prompt = f"请将以下文本改写为指定风格：\n{text}\n改写后："
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def extract_and_transfer(
        self,
        sample_texts: List[str],
        target_text: str,
        epochs: int = 5,
        rank: int = 8,
        learning_rate: float = 1e-4,
        progress_callback=None,
    ) -> dict:
        """
        Extract style from sample texts and apply to target text.
        This is the main pipeline: train LoRA on samples, then generate.
        """
        stats = self.train_style(
            sample_texts,
            epochs=epochs,
            learning_rate=learning_rate,
            progress_callback=progress_callback,
        )

        result = self.style_transfer(target_text)
        return {"stats": stats, "result": result}