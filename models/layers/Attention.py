import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from models.layers.model_utils import get_laplace_mat

# 定义注意力层
class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention_scores = torch.matmul(x, x.transpose(-2, -1))
        attention_scores = self.softmax(attention_scores)

        attended_values = torch.matmul(attention_scores, self.linear(x))

        return attended_values