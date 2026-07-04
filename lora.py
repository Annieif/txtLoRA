"""
Pure PyTorch LoRA (Low-Rank Adaptation) implementation.
No external dependencies beyond torch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA layer wrapping a linear layer.
    Adds low-rank adaptation matrices A and B.
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)

        base = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return base + lora_out * self.scaling

    def merge(self):
        """Merge LoRA weights into the original linear layer."""
        if not self.merged:
            self.linear.weight.data += (self.lora_A @ self.lora_B).T * self.scaling
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from the original linear layer."""
        if self.merged:
            self.linear.weight.data -= (self.lora_A @ self.lora_B).T * self.scaling
            self.merged = False

    def get_lora_state_dict(self) -> dict:
        """Get LoRA parameters for saving."""
        return {
            "lora_A": self.lora_A.data.clone(),
            "lora_B": self.lora_B.data.clone(),
            "rank": self.rank,
            "alpha": self.alpha,
            "in_features": self.linear.in_features,
            "out_features": self.linear.out_features,
        }

    def load_lora_state_dict(self, state_dict: dict):
        """Load LoRA parameters."""
        self.lora_A.data = state_dict["lora_A"].to(self.lora_A.device)
        self.lora_B.data = state_dict["lora_B"].to(self.lora_B.device)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> dict:
    """
    Apply LoRA to target linear layers in the model.
    Freezes ALL non-LoRA parameters so only LoRA weights are trained.
    Returns a dict mapping module names to LoRALinear wrappers.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_layers = {}

    # Step 1: Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Replace target Linear layers with LoRALinear
    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if any(t in child_name for t in target_modules):
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, child_name, lora_layer)
                    full_name = f"{module_name}.{child_name}" if module_name else child_name
                    lora_layers[full_name] = lora_layer

    return lora_layers


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract all LoRA weights from the model."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name] = module.get_lora_state_dict()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict):
    """Load LoRA weights into the model."""
    for name, lora_state in state_dict.items():
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if isinstance(module, LoRALinear):
            module.load_lora_state_dict(lora_state)


def count_lora_parameters(model: nn.Module) -> tuple:
    """Count total and trainable (LoRA) parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable