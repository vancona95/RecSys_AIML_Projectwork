from dataclasses import dataclass
from typing import List

import torch

from BaseModel import BaseModel, BaseConfig


@dataclass
class DeepCoNNConfig(BaseConfig):
    max_review_length: int
    word_dim: int  # the dimension of word embedding
    kernel_widths: List[int]  # the window sizes of convolutional kernel
    kernel_deep: int  # the number of convolutional kernels
    latent_factors: int
    fm_k: int


class ConvMaxLayer(torch.nn.Module):
    """
    The independent layer for user review and item review.
    """

    def __init__(self, config: DeepCoNNConfig):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.maxs = torch.nn.ModuleList()
        for width in config.kernel_widths:
            self.convs.append(torch.nn.Conv1d(
                in_channels=config.word_dim,
                out_channels=config.kernel_deep,
                kernel_size=width,
                stride=1))
            self.maxs.append(torch.nn.MaxPool1d(
                kernel_size=config.max_review_length - width + 1,
                stride=1))

        self.activation = torch.nn.ReLU()
        self.full_connect = torch.nn.Linear(config.kernel_deep * len(config.kernel_widths), config.latent_factors)

    def forward(self, review):
        """
        Input Shape: (BatchSize, ReviewLength, WordEmbeddingSize)
        Output Shape: (BatchSize, LatentFactorsSize)
        """

        outputs = []
        review = review.permute(0, 2, 1)
        for max_pool, conv in zip(self.maxs, self.convs):
            out = self.activation(conv(review))
            max_out = max_pool(out)
            flatten_out = torch.flatten(max_out, start_dim=1)
            outputs.append(flatten_out)

        conv_out = torch.cat(outputs, dim=1)
        latent = self.full_connect(conv_out)

        return latent


class FMLayer(torch.nn.Module):
    """
    The implementation of Factorization machine.
    Reference: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
    Input Shape: (BatchSize, LatentFactorsSize * 2)
    Output Shape: (BatchSize)
    """

    def __init__(self, config: DeepCoNNConfig):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(config.latent_factors * 2, config.fm_k), requires_grad=True)
        self.lin = torch.nn.Linear(config.latent_factors * 2, 1)

    def forward(self, x):
        s1_square = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        s2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)

        out_inter = 0.5 * (s1_square - s2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out


class DeepCoNN(BaseModel):
    """
    Main network, including two independent ConvMaxLayers and one shared FMLayer.
    """

    def __init__(self, config: DeepCoNNConfig, embedding_weight):
        assert config is not None
        super().__init__(config)

        self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        self.embedding.weight.requires_grad = False

        self.user_layer = ConvMaxLayer(config)
        self.item_layer = ConvMaxLayer(config)
        self.share_layer = FMLayer(config)

    def forward(self, user_review, item_review):
        """
        Input Shape: (BatchSize, ReviewLength)
        Output Shape: (BatchSize)
        """

        user_review = self.embedding(user_review)
        item_review = self.embedding(item_review)
        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)
        latent = torch.cat([user_latent, item_latent], dim=1)
        predict = self.share_layer(latent)
        return predict


