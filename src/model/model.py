from typing import Optional, Literal, List, Tuple

from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.model.llama import (
    LlamaModel,
    create_data_format,
    tokenize,
)
from src.model.moment import MOMENTPipeline, TimeseriesOutputs
from src.utils import freeze_parameters, unfreeze_parameters

LLM_BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TIMESERIES_BASE_MODEL_NAME = "AutonLab/MOMENT-1-large"


GENERATE_RESULT_PROMPT = """You are an expert time-series prediction system.
Your input is a natural-language report describing one or more time series.
Identify and leverage column (feature) details and inter-column relationships mentioned in the report to interpret patterns (trend, seasonality, anomalies, missing data, etc.), then produce a binary prediction based on the report's evidence.

1. Extract time granularity, key variables, units, aggregation rules, baselines/thresholds from the report.
2. If causal/correlational relationships between columns are described, use them as evidence.
3. Analyze this report and summarize the features for prediction."""


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
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))

        # Llama Model
        if self.llama_enabled:
            self.llama_model, self.llama_head = self.init_llama(
                llama_model_name, task_name=self.task_name
            )
            self.llama_model.to(self.device)
            # freeze_parameters(self.llama_model)
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            self.llama_tokenizer.padding_side = "left"
        # Moment Model
        if self.moment_enabled:
            self.moment_model, _ = self.init_moment(
                moment_model_name,
                task_name=self.task_name,
                num_channels=num_channels,
            )
            self.moment_model.to(self.device)

    @staticmethod
    def init_llama(
        llama_model_name: str,
        task_name: Literal["classification", "forecasting"] = "classification",
    ) -> Tuple[LlamaModel, Optional[nn.Linear]]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        llama_model = LlamaModel.from_pretrained(
            llama_model_name,
            quantization_config=bnb_config,
        )

        if task_name == "classification":
            llama_head = nn.Sequential(
                nn.Linear(
                    llama_model.config.hidden_size, llama_model.config.hidden_size
                ),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(llama_model.config.hidden_size, 2),
            )
        elif task_name == "forecasting":
            llama_head = nn.Sequential(
                nn.Linear(
                    llama_model.config.hidden_size, llama_model.config.hidden_size
                ),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(llama_model.config.hidden_size, 1),
            )
        else:
            llama_head = None

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
        return llama_model, llama_head

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

    def forward_llama(self, reports: List[str]) -> torch.Tensor:
        assert self.llama_enabled, "Llama model is not enabled."

        prompt = GENERATE_RESULT_PROMPT
        messages = [
            create_data_format(messages=[prompt, report], roles=["system", "user"])
            for report in reports
        ]

        kwargs = tokenize(
            tokenizer=self.llama_tokenizer,
            messages=messages,
            add_generation_prompt=True,
            device=self.device,
        )
        llama_emb = self.llama_model(**kwargs).last_hidden_state[:, -1, :]
        llama_outputs = self.llama_head(llama_emb)
        return llama_outputs

    def forward_moment(self, timeseries: torch.Tensor) -> torch.Tensor:
        assert self.moment_enabled, "Moment model is not enabled."
        # timeseries : batchsize, n_channels, context_length
        assert (
            timeseries.size(2) <= 512
        ), "Moment model supports maximum sequence length of 512."
        max_seq_len = 512
        pad = max_seq_len - timeseries.size(2)
        if pad > 0:
            timeseries = torch.nn.functional.pad(
                timeseries, (pad, 0), mode="constant", value=0.0
            )
        batch_size = timeseries.size(0)
        input_mask = torch.zeros(
            batch_size, max_seq_len, dtype=torch.bool, device=timeseries.device
        )
        if pad > 0:
            input_mask[:, pad:] = 1

        moment_outputs = self.moment_model(
            x_enc=timeseries,
            reduction=self.moment_reduction_method,
            input_mask=input_mask,
        )
        return moment_outputs

    def get_fusion_ratio(self):
        if self.training:
            assert self.fusion_weight.requires_grad
        alpha = torch.sigmoid(self.fusion_weight)
        clamped_val = 0.1 + (alpha * 0.8)
        return clamped_val

    def save(self, save_directory: str | Path):
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir(parents=True, exist_ok=True)
        if self.llama_enabled:
            torch.save(self.llama_head.state_dict(), save_directory / "llama_head.pth")
        if self.moment_enabled:
            torch.save(
                self.moment_model.head.state_dict(), save_directory / "moment_head.pth"
            )

    def load(self, load_directory: str | Path):
        if isinstance(load_directory, str):
            load_directory = Path(load_directory)
        if not load_directory.exists():
            raise ValueError(f"Load directory {load_directory} does not exist.")

        if self.llama_enabled:
            w = torch.load(load_directory / "llama_head.pth", map_location=self.device)
            self.llama_head.load_state_dict(w)
        if self.moment_enabled:
            w = torch.load(load_directory / "moment_head.pth", map_location=self.device)
            self.moment_model.head.load_state_dict(w)

    def print_num_trainable_parameters(self):
        print("Llama Trainable parameters:")
        llama_num_unfrozen_parameters = sum(
            p.numel() for p in self.llama_model.parameters() if p.requires_grad
        )
        llama_num_total_parameters = sum(
            p.numel() for p in self.llama_model.parameters()
        )
        print(
            f"Llama Model : {llama_num_unfrozen_parameters} ({llama_num_unfrozen_parameters / llama_num_total_parameters * 100:.2f} % of total)"
        )
        print(
            "Llama Head : ",
            sum(p.numel() for p in self.llama_head.parameters() if p.requires_grad),
        )
        print()
        print("Moment Trainable parameters:")

        moment_num_unfrozen_parameters = sum(
            p.numel() for p in self.moment_model.parameters() if p.requires_grad
        )
        moment_num_total_parameters = sum(
            p.numel() for p in self.moment_model.parameters()
        )
        print(
            f"Moment Model (include head) : {moment_num_unfrozen_parameters} ({moment_num_unfrozen_parameters / moment_num_total_parameters * 100:.2f} % of total)"
        )
