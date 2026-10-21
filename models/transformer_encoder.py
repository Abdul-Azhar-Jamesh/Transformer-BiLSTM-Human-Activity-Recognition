"""


MATH / ARCHITECTURE:
    1. x_emb = PatchEmbed(x)        -> converts raw window to tokens
    2. x_emb = PositionalEncode(x_emb)
    3. loop L times:
           x_emb = TransformerBlock(x_emb)
    4. x_out = FinalLayerNorm(x_emb)
    
    Final LayerNorm is required in Pre-LN architectures before sending
    the representation to the final head (classifier or MAE decoder).

TENSOR SHAPES:
    Input:       (B, T, C)
    Output:      (B, N_p, D)


"""

import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.positional_encoding import LearnablePositionalEncoding
from models.transformer_block import TransformerEncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int,
                 num_heads: int, num_layers: int, max_len: int, dropout_p: float = 0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embed = LearnablePositionalEncoding(max_len, embed_dim)
        
        # nn.ModuleList allows us to iterate through layers while keeping them registered
        # so PyTorch knows they are parameters of this model.
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout_p)
            for _ in range(num_layers)
        ])
        
        # Crucial for Pre-LN architecture: a final layernorm before the task head
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, return_all_attention: bool = False):
        """
        x: (B, T, C)
        Returns:
            out (B, N_p, D)
            [optionally] attn_maps (list of Tensors of shape (B, H, N_p, N_p))
        """
        # 1. Patch & Positional Embedding
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        
        # 2. Transformer Blocks
        attn_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_maps.append(attn_weights)
            
        # 3. Final LayerNorm
        x = self.norm(x)
        
        if return_all_attention:
            return x, attn_maps
        return x

    def forward_patched(self, x_patched: torch.Tensor, return_all_attention: bool = False):
        """
        FOR MAE PRETRAINING ONLY:
        Takes already-patched and masked tokens, skips patch_embed.
        x_patched: (B, N_visible, D)
        """
        x = self.pos_embed(x_patched)
        
        attn_maps = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_maps.append(attn_weights)
            
        x = self.norm(x)
        
        if return_all_attention:
            return x, attn_maps
        return x


if __name__ == "__main__":
    # Self-test (UCI HAR configuration)
    B, T, C = 32, 128, 6
    P = 16
    D = 128
    
    encoder = TransformerEncoder(
        in_channels=C, patch_size=P, embed_dim=D,
        num_heads=4, num_layers=4, max_len=8
    )
    
    x_test = torch.randn(B, T, C)
    out, maps = encoder(x_test, return_all_attention=True)
    
    print("TRANSFORMER ENCODER SET TEST")
    print(f"  Input : {tuple(x_test.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Attention maps captured: {len(maps)} (one for each layer)")
    assert out.shape == (B, 8, 128)
    assert len(maps) == 4
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total Params: {total_params:,d}")
    print("  Status : PASS")
