import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationModule(nn.Module):
    def __init__(self, dim=256):
        super(RelationModule, self).__init__()
        self.query_proj = nn.Linear(dim, dim)  # Linear projection for Query
        self.key_proj = nn.Linear(dim, dim)    # Linear projection for Key
        self.scale = dim ** -0.5  # Scaled factor for attention scores

        self.layer_norm_feat = nn.LayerNorm(dim)  # LayerNorm for feat
        self.layer_norm_mask = nn.LayerNorm(dim)  # LayerNorm for mask

    def forward(self, feat, mask):
        # Apply LayerNorm to feat and mask (pre-norm)
        feat = self.layer_norm_feat(feat)  # (bsz, h*w, dim)
        mask = self.layer_norm_mask(mask)  # (bsz, 1, dim)

        # Linear projections for Query and Key
        query = self.query_proj(mask)  # (bsz, 1, dim)
        key = self.key_proj(feat)      # (bsz, h*w, dim)

        # Compute scaled attention scores
        linear_attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale  # (bsz, 1, h*w)

        # Residual attention scores (direct dot product)
        residual_attn_scores = torch.matmul(mask, feat.transpose(-1, -2))  # (bsz, 1, h*w)

        # Combine linear and residual attention scores
        attn_scores = linear_attn_scores + residual_attn_scores
        
        attn_scores = attn_scores.squeeze(1)
        
        batch_size = attn_scores.shape[0]
        attn_score_norm = F.normalize(attn_scores.view(batch_size,-1))

        return attn_score_norm

