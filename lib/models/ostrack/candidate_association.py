"""Small frozen-feature candidate association head used by the B2 experiment."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class CandidateAssociationNet(nn.Module):
    """Associate a previous target candidate with current Top-K candidates.

    The OSTrack backbone is intentionally external and frozen.  To keep the
    trajectory cache compact, raw 768-D shallow/deep tokens are compressed by
    fixed Gaussian projections before the learnable encoders.  The projections
    are registered buffers, so collection, training, and inference always use
    exactly the same mapping.
    """

    def __init__(self, token_dim=768, compressed_dim=128, hidden_dim=96,
                 embedding_dim=96, geometry_dim=5, temperature=0.10):
        super().__init__()
        self.token_dim = token_dim
        self.compressed_dim = compressed_dim
        self.temperature = temperature

        generator = torch.Generator(device='cpu')
        generator.manual_seed(20260711)
        scale = 1.0 / math.sqrt(compressed_dim)
        self.register_buffer('deep_projection', torch.randn(token_dim, compressed_dim, generator=generator) * scale)
        self.register_buffer('shallow_projection', torch.randn(token_dim, compressed_dim, generator=generator) * scale)

        self.deep_encoder = nn.Sequential(nn.Linear(compressed_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.shallow_encoder = nn.Sequential(nn.Linear(compressed_dim, hidden_dim), nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.geometry_encoder = nn.Sequential(nn.Linear(geometry_dim, hidden_dim // 2), nn.ReLU(),
                                              nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.ReLU())
        self.embedding = nn.Sequential(nn.Linear(hidden_dim * 2 + hidden_dim // 2, embedding_dim), nn.ReLU(),
                                       nn.Linear(embedding_dim, embedding_dim))
        self.candidate_bias = nn.Sequential(nn.Linear(geometry_dim, hidden_dim // 2), nn.ReLU(),
                                            nn.Linear(hidden_dim // 2, 1))
        self.dustbin = nn.Sequential(nn.Linear(embedding_dim, hidden_dim // 2), nn.ReLU(),
                                     nn.Linear(hidden_dim // 2, 1))

    def compress(self, deep, shallow):
        """Apply fixed projection to raw candidate tokens."""
        return deep @ self.deep_projection, shallow @ self.shallow_projection

    def encode_compressed(self, deep, shallow, geometry):
        deep_feature = self.deep_encoder(deep.float())
        shallow_feature = self.shallow_encoder(shallow.float())
        geometry_feature = self.geometry_encoder(geometry.float())
        embedding = self.embedding(torch.cat((deep_feature, shallow_feature, geometry_feature), dim=-1))
        return F.normalize(embedding, dim=-1)

    def encode_raw(self, deep, shallow, geometry):
        deep, shallow = self.compress(deep, shallow)
        return self.encode_compressed(deep, shallow, geometry)

    def logits(self, previous_embedding, current_embedding, current_geometry):
        """Return K association logits followed by one dustbin logit."""
        if previous_embedding.dim() == 1:
            previous_embedding = previous_embedding.unsqueeze(0)
        if current_embedding.dim() == 2:
            current_embedding = current_embedding.unsqueeze(0)
            current_geometry = current_geometry.unsqueeze(0)
        pair_logits = (previous_embedding.unsqueeze(1) * current_embedding).sum(-1) / self.temperature
        pair_logits = pair_logits + self.candidate_bias(current_geometry.float()).squeeze(-1)
        dustbin_logit = self.dustbin(previous_embedding).squeeze(-1).unsqueeze(-1)
        return torch.cat((pair_logits, dustbin_logit), dim=-1)
