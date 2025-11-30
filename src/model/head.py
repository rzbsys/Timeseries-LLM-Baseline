import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        n_classes: int,
        llama_output_dim: int,
        moment_output_dim: int,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.input_dim = llama_output_dim + moment_output_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(
                self.input_dim,
                n_classes,
            ),
        )

    def forward(self, llama_emb: torch.Tensor, moment_emb: torch.Tensor) -> torch.Tensor:
        context = torch.cat([llama_emb, moment_emb], dim=1)
        logits = self.classifier(context)
        return logits


class TRMHead(nn.Module):
    def __init__(
        self,
        n_classes: int,
        llama_output_dim: int,
        moment_output_dim: int,
        trm_dim: int = 512,
        n_recursion: int = 6,
        n_loops: int = 3,
    ):
        super().__init__()
        self.trm_dim = trm_dim
        self.n_recursion = n_recursion
        self.n_classes = n_classes
        self.n_loops = n_loops
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
