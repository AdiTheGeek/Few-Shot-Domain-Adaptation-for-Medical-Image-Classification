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

        # Monkey-patch forward_features if possible
        orig_ff = vit_model.forward_features

        def patched_forward_features(x):
            x = vit_model.patch_embed(x)
            cls_token = vit_model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if hasattr(vit_model, 'visual_prompt'):
                x = vit_model.visual_prompt(x)
            x = x + vit_model.pos_embed
            x = vit_model.pos_drop(x)
            for blk in vit_model.blocks:
                x = blk(x)
            x = vit_model.norm(x)
            return x[:, 0]

        vit_model.forward_features = patched_forward_features
    return vit_model
