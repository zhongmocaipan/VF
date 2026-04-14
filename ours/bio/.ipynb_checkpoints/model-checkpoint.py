import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import math
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix

# ======================== MultiCNN + BiLSTM ========================
class MultiCNN_BiLSTM(nn.Module):
    def __init__(self):
        super(MultiCNN_BiLSTM, self).__init__()
        self.filter_sizes = [1, 2, 3, 4, 5, 6]
        filter_num = 32
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (fsz, 25)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)
        
        self.lstm = nn.LSTM(
            input_size=len(self.filter_sizes) * filter_num,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [F.max_pool2d(item, kernel_size=(item.size(2), item.size(3))) for item in conv_outputs]
        pooled_outputs = [item.view(item.size(0), -1) for item in pooled_outputs]
        cnn_feature = torch.cat(pooled_outputs, dim=1)

        lstm_in = cnn_feature.view(cnn_feature.size(0), 1, -1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feature = lstm_out[:, -1, :]
        lstm_feature = self.dropout(lstm_feature)
        output = self.block1(lstm_feature)
        return output

class MLPBranch(nn.Module):
    def __init__(self):
        super(MLPBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        return self.mlp(x)

# ======================== BAN 双线性注意力融合 ========================
class BAN(nn.Module):
    def __init__(self, dim=64, hidden_dim=64):
        super().__init__()
        self.proj1 = nn.Linear(dim, hidden_dim)
        self.proj2 = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat1, feat2):
        a = self.proj1(feat1)
        b = self.proj2(feat2)
        attn = torch.sigmoid(torch.sum(a * b, dim=-1, keepdim=True))
        fused = attn * feat1 + (1 - attn) * feat2
        fused = self.norm(fused + self.out_proj(fused))
        return fused

# ======================== 主模型：完全稳定版 ========================
class FusionPepNet(nn.Module):
    def __init__(self):
        super(FusionPepNet, self).__init__()
        self.branch1 = MultiCNN_BiLSTM()
        self.branch2 = MLPBranch()
        self.fusion_attn = BAN(dim=64)
        
        # 稳定MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, features):
        feat1 = self.branch1(input_ids)
        feat2 = self.branch2(features)
        fused = self.fusion_attn(feat1, feat2)
        logits = self.classifier(fused)
        return logits, feat1, feat2

# ======================== HSIC 损失 ========================
def hsic_loss(X, Y):
    N = X.size(0)
    K = torch.matmul(X, X.t())
    R = torch.matmul(Y, Y.t())
    H = torch.eye(N, device=X.device) - (1.0 / N) * torch.ones((N, N), device=X.device)
    hsic = torch.trace(torch.matmul(torch.matmul(K, H), torch.matmul(R, H))) / (N * N)
    return hsic