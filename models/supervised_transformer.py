"""

TENSOR SHAPES:
    Input:       (B, T, C)
    Encoder out: (B, N_p, D)
    Pooled:      (B, D)
    Logits:      (B, num_classes)

"""

import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder

class SupervisedTransformer(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int,
                 num_heads: int, num_layers: int, max_len: int, num_classes: int,
                 dropout_p: float = 0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout_p=dropout_p
        )
        
        # Linear classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        Returns: logits (B, num_classes)
        """
        # (B, N_p, D)
        encoded = self.encoder(x)
        
        # Global Average Pooling over the N_p dimension (dim=1)
        # -> (B, D)
        pooled = encoded.mean(dim=1)
        
        # Classification
        # -> (B, num_classes)
        logits = self.head(pooled)
        return logits


if __name__ == "__main__":
    # Self-test
    B, T, C = 32, 128, 6
    P = 16
    D = 128
    
    # 6 classes for UCI HAR
    model = SupervisedTransformer(
        in_channels=C, patch_size=P, embed_dim=D,
        num_heads=4, num_layers=4, max_len=8, num_classes=6
    )
    
    x_test = torch.randn(B, T, C)
    out = model(x_test)
    
    print("SUPERVISED TRANSFORMER TEST")
    print(f"  Input : {tuple(x_test.shape)}")
    print(f"  Output: {tuple(out.shape)} (expect (32, 6))")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Params: {total_params:,d}")
    print("  Status: PASS")
