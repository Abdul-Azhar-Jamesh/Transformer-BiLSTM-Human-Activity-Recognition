"""


TENSOR SHAPES :
    Input:       (B, 128, 6)   (T=128 timesteps, C=6 channels)
    Patch size:  P=16
    Reshaped:    (B, 8, 16, 6)
    Flattened:   (B, 8, 96)    (16 * 6 = 96 raw features per patch)
    Projected:   (B, 8, 128)   (if Embedding dim D = 128)


"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        """
        Args:
            in_channels (int): Number of raw sensors (e.g. 6 for UCI, 9 for SisFall)
            patch_size (int): Number of timesteps per patch (e.g. P=16)
            embed_dim (int): Target embedding dimension (e.g. D=128)
        """
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Linear projection from raw patch features to embedding space
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        Returns: (B, N_p, D)
        """
        B, T, C = x.shape
        
        # Ensure T is perfectly divisible by patch_size
        assert T % self.patch_size == 0, f"Sequence length {T} must be divisible by patch size {self.patch_size}."
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}."
        
        N_p = T // self.patch_size
        
        # 1. Reshape to (B, N_p, P, C)
        x = x.view(B, N_p, self.patch_size, C)
        
        # 2. Flatten P and C -> (B, N_p, P*C)
        x = x.view(B, N_p, self.patch_size * C)
        
        # 3. Project -> (B, N_p, D)
        x = self.proj(x)
        
        return x


if __name__ == "__main__":
    # Self-test: UCI HAR configuration
    B, T, C = 32, 128, 6
    P = 16
    D = 128
    
    model = PatchEmbedding(in_channels=C, patch_size=P, embed_dim=D)
    x_test = torch.randn(B, T, C)
    out = model(x_test)
    
    print("PATCH EMBEDDING TEST (UCI HAR scale)")
    print(f"  Input  : {tuple(x_test.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (B, 8, 128), "Shape mismatch!"
    print("  Status : PASS")
