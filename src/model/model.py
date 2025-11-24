from typing import Optional, Literal, List, Tuple, Dict
from dataclasses import dataclass

from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.model.llama import LlamaModel
from src.model.moment import MOMENTPipeline
from src.utils import freeze_parameters


@dataclass
class LlamaOutputs:
    last_hidden_state: torch.Tensor = None


@dataclass
class MomentOutputs:
    last_hidden_state: torch.Tensor = None


def init_llama(llama_model_name: str, quantization: bool = False, model_compile: bool = False, init_lora: bool = False) -> LlamaModel:
    model_config = {
        # "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_config["quantization_config"] = bnb_config

    llama_model = LlamaModel.from_pretrained(llama_model_name, **bnb_config)
    if model_compile:
        llama_model = torch.compile(llama_model, mode="max-autotune")
    # Lora
    if init_lora:
        peft_config = LoraConfig(
            inference_mode=False,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules=["q_proj", "k_proj"],
        )
        llama_model = prepare_model_for_kbit_training(llama_model)
        llama_model = get_peft_model(llama_model, peft_config)
    else:
        freeze_parameters(llama_model)
    return llama_model


def init_moment(moment_model_name: str) -> MOMENTPipeline:
    config = {"task_name": "embedding"}

    moment_model = MOMENTPipeline.from_pretrained(
        moment_model_name,
        model_kwargs=config,
    )
    moment_model.init()
    return moment_model


class TRMModule(nn.Module):
    def __init__(
        self,
        n_classes: int,
        llama_output_dim: int,
        moment_output_dim: int,
        trm_dim: int = 512,
        n_recursion: int = 6,
    ):
        super().__init__()
        self.trm_dim = trm_dim
        self.n_recursion = n_recursion
        self.n_classes = n_classes
        self.z_init = nn.Parameter(torch.randn(1, 1, trm_dim))

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=trm_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )  # Pre-Norm for stability
        self.llama_proj = nn.Linear(llama_output_dim, trm_dim)
        self.moment_proj = nn.Linear(moment_output_dim, trm_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(trm_dim),
            nn.Linear(
                trm_dim,
                n_classes,
            ),
        )

    def forward_recursion_step(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        new_z = self.decoder_layer(z, context)
        return new_z

    def forward(self, llama_emb: torch.Tensor, moment_emb: torch.Tensor) -> torch.Tensor:
        llama_context = self.llama_proj(llama_emb).unsqueeze(1)
        moment_context = self.moment_proj(moment_emb).unsqueeze(1)

        # Combine contexts
        context = torch.cat([llama_context, moment_context], dim=1)

        # Initialize z
        batch_size = context.size(0)
        z = self.z_init.expand(batch_size, -1, -1)

        # Recursion
        with torch.no_grad():
            for _ in range(self.n_loops - 1):
                for _ in range(self.n_recursion):
                    z = self.forward_recursion_step(z, context)

        z = z.detach()
        for _ in range(self.n_recursion):
            z = self.forward_recursion_step(z, context)

        # Classification
        logits = self.classifier(z.squeeze(1))
        return logits


class TRMFusionBTSModel(nn.Module):
    def __init__(
        self,
        llama_model_name: str,
        moment_model_name: str,
        n_classes: int,
        trm_dim: int = 512,
        n_recursion: int = 6,
        n_loops: int = 3,
    ):
        super().__init__()
        self.trm_dim = trm_dim
        self.n_recursion = n_recursion
        self.n_loops = n_loops
        self.n_classes = n_classes
        self.llama_model_name = llama_model_name
        self.moment_model_name = moment_model_name

        llama_model = init_llama(self.llama_model_name)
        moment_model = init_moment(self.moment_model_name)

        llama_output_dim = self.llama_model.config.hidden_size
        moment_output_dim = self.moment_model.config.d_model
        self.trm_module = TRMModule(
            n_classes=self.n_classes,
            llama_output_dim=llama_output_dim,
            moment_output_dim=moment_output_dim,
            trm_dim=trm_dim,
            n_recursion=n_recursion,
        )

        self.register_buffer("llama_model", llama_model, persistent=False)
        self.register_buffer("moment_model", moment_model, persistent=False)

    def forward_llama(self, llama_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        llama_emb = self.llama_model(**llama_inputs).last_hidden_state[:, -1, :]
        return LlamaOutputs(
            last_hidden_state=llama_emb,
        )

    def forward_moment(self, timeseries: torch.Tensor) -> torch.Tensor:
        # timeseries : batchsize, n_channels, context_length
        assert timeseries.size(2) <= 512, "Moment model supports maximum sequence length of 512."
        max_seq_len = 512
        pad = max_seq_len - timeseries.size(2)
        if pad > 0:
            timeseries = torch.nn.functional.pad(timeseries, (pad, 0), mode="constant", value=0.0)
        batch_size = timeseries.size(0)
        input_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=timeseries.device)
        if pad > 0:
            input_mask[:, pad:] = 1

        moment_outputs = self.moment_model.embed(
            x_enc=timeseries,
            reduction="mean",
            input_mask=input_mask,
        )

        return MomentOutputs(last_hidden_state=moment_outputs.embeddings)

    def forward(self, llama_inputs: Dict[str, torch.Tensor], timeseries: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            llama_outputs = self.forward_llama(llama_inputs)
            moment_outputs = self.forward_moment(timeseries)
        logits = self.trm_module(llama_outputs.last_hidden_state, moment_outputs.last_hidden_state)
        return logits

    def save(self, save_directory: str | Path):
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.trm_module.state_dict(), save_directory / "trm_head.pth")

    def load(self, load_directory: str | Path):
        if isinstance(load_directory, str):
            load_directory = Path(load_directory)
        if not load_directory.exists():
            raise ValueError(f"Load directory {load_directory} does not exist.")
        self.trm_module.load_state_dict(torch.load(load_directory / "trm_head.pth", map_location="cpu"))

    def print_num_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {total_params - trainable_params}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
