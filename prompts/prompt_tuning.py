import torch
import torch.nn as nn


class VisualPrompt(nn.Module):
    """Learnable prompt tokens prepended to patch embeddings for ViT.
    Works by overriding the `forward_features` embedding if using timm ViT.
    """
    def __init__(self, prompt_tokens: int, embed_dim: int):
        super().__init__()
        self.prompt_tokens = prompt_tokens
        self.embed_dim = embed_dim
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_tokens, embed_dim))

    def forward(self, x):
        # x shape: [B, N, D] embeddings (including cls token)
        b = x.shape[0]
        prompts = self.prompt_embeddings.unsqueeze(0).expand(b, -1, -1)
        # insert prompts after class token (assumes x[:,0] is cls)
        cls_tok = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        x = torch.cat([cls_tok, prompts, patch_tokens], dim=1)
        return x


def attach_visual_prompt_to_vit(vit_model: nn.Module, prompt_tokens: int = 10):
    # This is a best-effort attachment for timm ViT: override patch embedding forward
    if hasattr(vit_model, 'patch_embed') and hasattr(vit_model, 'pos_embed'):
        embed_dim = vit_model.patch_embed.proj.out_channels if hasattr(vit_model.patch_embed, 'proj') else vit_model.embed_dim
        vp = VisualPrompt(prompt_tokens, embed_dim)
        vit_model.visual_prompt = vp

        # Store reference to avoid closure issues
        _vit_model = vit_model
        
        # Monkey-patch the full forward method (not just forward_features)
        # because timm's forward() does more than just call forward_features() + forward_head()
        def patched_forward(x, attn_mask=None):
            # Patch embedding
            x = _vit_model.patch_embed(x)
            # Prepend class token
            cls_token = _vit_model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            # Add positional embeddings BEFORE adding prompts
            x = x + _vit_model.pos_embed
            # Now add visual prompts (which don't need positional embeddings)
            if hasattr(_vit_model, 'visual_prompt'):
                x = _vit_model.visual_prompt(x)
            x = _vit_model.pos_drop(x)
            # Pass through transformer blocks
            for blk in _vit_model.blocks:
                x = blk(x)
            x = _vit_model.norm(x)
            # Return CLS token features (this is what the ViTWrapper expects)
            return x[:, 0]

        vit_model.forward = patched_forward
    return vit_model
