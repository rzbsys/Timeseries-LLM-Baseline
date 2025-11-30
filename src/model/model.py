from typing import Optional, Literal, List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.model.llama import LlamaModel
from src.model.moment import MOMENTPipeline
from src.utils import freeze_parameters

from .head import MLPHead, TRMHead


@dataclass
class LlamaOutputs:
    representation: torch.Tensor = None


@dataclass
class MomentOutputs:
    representation: torch.Tensor = None


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

    llama_model = LlamaModel.from_pretrained(llama_model_name, **model_config)
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
        print("Initialized Llama model with LoRA.")
        llama_model.print_trainable_parameters()

    else:
        freeze_parameters(llama_model)
    return llama_model


def init_moment(moment_model_name: str, init_lora: bool = False) -> MOMENTPipeline:
    config = {"task_name": "embedding"}

    moment_model = MOMENTPipeline.from_pretrained(
        moment_model_name,
        model_kwargs=config,
    )
    moment_model.init()

    if init_lora:
        if hasattr(moment_model, "config") and not hasattr(moment_model.config, "get"):

            def config_get(key, default=None):
                return getattr(moment_model.config, key, default)

            moment_model.config.get = config_get

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # 또는 SEQ_2_SEQ_LM 등 모델 구조에 맞게 설정 (시계열은 보통 커스텀이므로 None이나 FEATURE_EXTRACTION 권장)
            inference_mode=False,
            r=8,  # Rank
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1,
            target_modules=["q", "v"],  # 위에서 확인한 타겟 모듈 지정
        )
        moment_model = get_peft_model(moment_model, peft_config)
        print("Initialized MOMENT model with LoRA.")
        moment_model.print_trainable_parameters()

    return moment_model


class BTSModel(nn.Module):
    def __init__(
        self,
        llama_model_name: str,
        moment_model_name: str,
        n_classes: int,
        n_channels: int,
        head_type: Literal["trm", "mlp"],
        head_configs: Optional[Dict] = {},
        # trm_dim: int = 512,
        # n_recursion: int = 6,
        # n_loops: int = 3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.llama_model_name = llama_model_name
        self.moment_model_name = moment_model_name
        self.head_type = head_type

        self.llama_model = init_llama(self.llama_model_name, quantization=True, init_lora=True)
        self.moment_model = init_moment(self.moment_model_name, init_lora=True)

        llama_output_dim = self.llama_model.config.hidden_size
        # concat all channel embeddings
        moment_output_dim = self.moment_model.config.d_model * n_channels

        if head_type == "trm":
            self.head = TRMHead(
                n_classes=self.n_classes,
                llama_output_dim=llama_output_dim,
                moment_output_dim=moment_output_dim,
                **head_configs,
            )
        elif head_type == "mlp":
            self.head = MLPHead(
                n_classes=self.n_classes,
                llama_output_dim=llama_output_dim,
                moment_output_dim=moment_output_dim,
                **head_configs,
            )
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward_llama(self, llama_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        llama_emb = self.llama_model(**llama_inputs).last_hidden_state[:, -1, :]
        return LlamaOutputs(
            representation=llama_emb,
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
            reduction="concat",
            input_mask=input_mask,
        )

        return MomentOutputs(representation=moment_outputs.embeddings)

    def forward(self, llama_inputs: Dict[str, torch.Tensor], timeseries: torch.Tensor) -> torch.Tensor:
        llama_outputs = self.forward_llama(llama_inputs)
        moment_outputs = self.forward_moment(timeseries)
        logits = self.head(llama_outputs.representation, moment_outputs.representation)
        return logits

    def save(self, save_directory: str | Path):
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), save_directory / f"{self.head_type}_head.pth")

        if hasattr(self.llama_model, "save_pretrained"):
            self.llama_model.save_pretrained(save_directory / "llama_model")
        if hasattr(self.moment_model, "save_pretrained"):
            self.moment_model.save_pretrained(save_directory / "moment_model")

    def load(self, load_directory: str | Path):
        if isinstance(load_directory, str):
            load_directory = Path(load_directory)
        if not load_directory.exists():
            raise ValueError(f"Load directory {load_directory} does not exist.")
        self.head.load_state_dict(torch.load(load_directory / f"{self.head_type}_head.pth", map_location="cpu"))
        if hasattr(self.llama_model, "load_pretrained"):
            self.llama_model = self.llama_model.from_pretrained(load_directory / "llama_model")
        if hasattr(self.moment_model, "load_pretrained"):
            self.moment_model = self.moment_model.from_pretrained(load_directory / "moment_model")

    # def print_num_trainable_parameters(self):
    #     total_params = sum(p.numel() for p in self.parameters())
    #     trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     print(f"Total parameters: {total_params}")
    #     print(f"Trainable parameters: {trainable_params}")
    #     print(f"Non-trainable parameters: {total_params - trainable_params}")
    #     print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
