from typing import Optional, Literal, List, Tuple, Dict
from dataclasses import dataclass

from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.model.llama import (
    LlamaModel,
    create_data_format,
    tokenize,
)
from src.model.moment import MOMENTPipeline, TimeseriesOutputs
from src.utils import freeze_parameters, unfreeze_parameters

LLM_BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TIMESERIES_BASE_MODEL_NAME = "AutonLab/MOMENT-1-large"


@dataclass
class LlamaOutputs:
    last_hidden_state: torch.Tensor = None


@dataclass
class MomentOutputs:
    last_hidden_state: torch.Tensor = None


class CombineHead(nn.Module):
    def __init__(self, llama_input_dim: int, moment_input_dim: int, n_classes: int):
        super().__init__()
        self.llama_fusion_net = nn.Linear(llama_input_dim, 128)
        self.moment_fusion_net = nn.Linear(moment_input_dim, 128)
        self.combine_nn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, llama_outputs: LlamaOutputs, moment_outputs: MomentOutputs) -> torch.Tensor:
        llama_feat = self.llama_fusion_net(llama_outputs.last_hidden_state)
        moment_feat = self.moment_fusion_net(moment_outputs.last_hidden_state)
        combined_feat = torch.cat([llama_feat, moment_feat], dim=-1)
        return self.combine_nn(combined_feat)


class BTSModel(nn.Module):
    def __init__(
        self,
        num_channels: Optional[int] = None,
        llama_model_name: Optional[str] = LLM_BASE_MODEL_NAME,
        moment_model_name: Optional[str] = TIMESERIES_BASE_MODEL_NAME,
        device: Optional[torch.device] = None,
        task_name: Literal["classification", "forecasting"] = "classification",
        moment_reduction_method: Literal["mean", "concat"] = "concat",
    ):
        super().__init__()
        self.llama_enabled = llama_model_name is not None
        self.moment_enabled = moment_model_name is not None
        self.device = device if device is not None else torch.device("cpu")
        self.task_name = task_name
        self.moment_reduction_method = moment_reduction_method

        # Llama Model
        if self.llama_enabled:
            self.llama_model, _ = self.init_llama(llama_model_name)
            self.llama_model.to(self.device)
        # Moment Model
        if self.moment_enabled:
            self.moment_model, _ = self.init_moment(
                moment_model_name,
                task_name=self.task_name,
                num_channels=num_channels,
            )
            self.moment_model.to(self.device)

        # for combine
        # self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        n_classes = 2 if task_name == "classification" else 1

        self.combind_head = CombineHead(
            llama_input_dim=self.llama_model.config.hidden_size,
            moment_input_dim=self.moment_model.config.d_model,
            n_classes=n_classes,
        ).to(self.device)

    @staticmethod
    def init_llama(
        llama_model_name: str,
        # task_name: Literal["classification", "forecasting"] = "classification",
    ) -> Tuple[LlamaModel, Optional[nn.Linear]]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        llama_model = LlamaModel.from_pretrained(
            llama_model_name,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        llama_model = torch.compile(llama_model, mode="max-autotune")

        # Lora
        peft_config = LoraConfig(
            inference_mode=False,
            lora_alpha=16,  # LoRA 스케일링 팩터
            lora_dropout=0.1,  # LoRA 드롭아웃 비율
            r=64,  # LoRA 랭크
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules=["q_proj", "k_proj"],
        )
        llama_model = prepare_model_for_kbit_training(llama_model)
        llama_model = get_peft_model(llama_model, peft_config)
        freeze_parameters(llama_model)
        return llama_model, None

    @staticmethod
    def init_moment(
        moment_model_name: str,
        task_name: Literal["classification", "forecasting"] = "classification",
        num_channels: Optional[int] = None,
    ) -> Tuple[MOMENTPipeline, None]:
        config = {
            "task_name": task_name,
            "freeze_embedder": True,
            "freeze_encoder": True,
            "freeze_head": False,
        }
        if task_name == "classification":
            config["num_class"] = 2
            if num_channels is not None:
                config["n_channels"] = num_channels
        elif task_name == "forecasting":
            config["forecast_horizon"] = 1
            config["head_dropout"] = 0.1
            config["weight_decay"] = 0

        moment_model = MOMENTPipeline.from_pretrained(
            moment_model_name,
            model_kwargs=config,
        )
        moment_model.init()
        unfreeze_parameters(moment_model.head)
        return moment_model, None

    def forward_llama(self, llama_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.llama_enabled, "Llama model is not enabled."
        with torch.no_grad():
            llama_emb = self.llama_model(**llama_inputs).last_hidden_state[:, -1, :]
        return LlamaOutputs(
            last_hidden_state=llama_emb,
        )

    def forward_moment(self, timeseries: torch.Tensor) -> torch.Tensor:
        assert self.moment_enabled, "Moment model is not enabled."
        # timeseries : batchsize, n_channels, context_length
        assert timeseries.size(2) <= 512, "Moment model supports maximum sequence length of 512."
        # print("device", timeseries.device)

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

    def combine_outputs(
        self,
        llama_outputs: LlamaOutputs,
        moment_outputs: MomentOutputs,
    ) -> torch.Tensor:
        llama_feat = self.combind_head(llama_outputs, moment_outputs)
        return llama_feat

    def save(self, save_directory: str | Path):
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.combind_head.state_dict(), save_directory / "combine_head.pth")
        self.llama_model.save_pretrained(save_directory / "llama_peft_model")

    def load(self, load_directory: str | Path):
        if isinstance(load_directory, str):
            load_directory = Path(load_directory)
        if not load_directory.exists():
            raise ValueError(f"Load directory {load_directory} does not exist.")
        self.combind_head.load_state_dict(torch.load(load_directory / "combine_head.pth", map_location=self.device))
        llama_adapter_dir = load_directory / "llama_peft_model"
        peft_config = PeftConfig.from_pretrained(llama_adapter_dir)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_llama = LlamaModel.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        base_llama = prepare_model_for_kbit_training(base_llama)
        self.llama_model = PeftModel.from_pretrained(
            base_llama,
            llama_adapter_dir,
            is_trainable=False,
        )
        self.llama_model.to(self.device)

    def print_num_trainable_parameters(self):
        print("Llama Trainable parameters:")
        llama_num_unfrozen_parameters = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
        llama_num_total_parameters = sum(p.numel() for p in self.llama_model.parameters())
        print(f"Llama Model : {llama_num_unfrozen_parameters} ({llama_num_unfrozen_parameters / llama_num_total_parameters * 100:.2f} % of total)")
        print()
        print("Moment Trainable parameters:")
        moment_num_unfrozen_parameters = sum(p.numel() for p in self.moment_model.parameters() if p.requires_grad)
        moment_num_total_parameters = sum(p.numel() for p in self.moment_model.parameters())
        print(f"Moment Model (include head) : {moment_num_unfrozen_parameters} ({moment_num_unfrozen_parameters / moment_num_total_parameters * 100:.2f} % of total)")
