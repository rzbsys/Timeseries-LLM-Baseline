from typing import Optional, Literal, List, Tuple

from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer

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


- Guidelines
Extract time granularity, key variables, units, aggregation rules, baselines/thresholds from the report.
If causal/correlational relationships between columns are described, use them as evidence."""

CLASSIFICATION_PROMPT = """
Output format:
0.23
"""

FORCASTING_PROMPT = """
Output format:
0.23
"""


class BTSModel(nn.Module):
    def __init__(
        self,
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
            self.llama_model, self.llama_head = self.init_llama(llama_model_name, task_name=self.task_name)
            self.llama_model.to(self.device)
            freeze_parameters(self.llama_model)
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        # Moment Model
        if self.moment_enabled:
            self.moment_model, _ = self.init_moment(
                moment_model_name,
                task_name=self.task_name,
            )
            self.moment_model.to(self.device)

    @staticmethod
    def init_llama(
        llama_model_name: str,
        task_name: Literal["classification", "forecasting"] = "classification",
    ) -> Tuple[LlamaModel, Optional[nn.Linear]]:
        llama_model = LlamaModel.from_pretrained(llama_model_name)
        if task_name == "classification":
            llama_head = nn.Sequential(
                nn.Linear(llama_model.config.hidden_size, llama_model.config.hidden_size),
                nn.Linear(llama_model.config.hidden_size, llama_model.config.hidden_size),
                nn.Linear(llama_model.config.hidden_size, 2)
            )
        elif task_name == "forecasting":
            llama_head = nn.Sequential(
                nn.Linear(llama_model.config.hidden_size, llama_model.config.hidden_size),
                nn.Linear(llama_model.config.hidden_size, llama_model.config.hidden_size),
                nn.Linear(llama_model.config.hidden_size, 1)
            )
        else:
            llama_head = None
        return llama_model, llama_head

    @staticmethod
    def init_moment(
        moment_model_name: str,
        task_name: Literal["classification", "forecasting"] = "classification",
    ) -> Tuple[MOMENTPipeline, None]:
        config = {
            "task_name": task_name,
            "freeze_embedder": True,
            "freeze_encoder": True,
            "freeze_head": False,
        }
        if task_name == "classification":
            config["num_class"] = 2
            config["n_channels"] = 1
        elif task_name == "forecasting":
            config["forecast_horizon"] = 1

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
        if self.task_name == "classification":
            prompt += CLASSIFICATION_PROMPT
        elif self.task_name == "forecasting":
            prompt += FORCASTING_PROMPT

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
        max_seq_len = 512
        pad = max_seq_len - timeseries.size(2)
        if pad > 0:
            timeseries = torch.nn.functional.pad(timeseries, (pad, 0), mode='constant', value=0.0)
        batch_size = timeseries.size(0)
        input_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=timeseries.device)
        if pad > 0:
            input_mask[:, pad:] = 1
        moment_outputs = self.moment_model(
            x_enc=timeseries,
            reduction=self.moment_reduction_method,
            input_mask=input_mask,
        )
        return moment_outputs

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
        print("Llama Model : ", sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad))
        print("Llama Head : ", sum(p.numel() for p in self.llama_head.parameters() if p.requires_grad))
        print()
        print("Moment Trainable parameters:")
        print("Moment Model : ", sum(p.numel() for p in self.moment_model.encoder.parameters() if p.requires_grad))
        print("Moment Head : ", sum(p.numel() for p in self.moment_model.head.parameters() if p.requires_grad))
