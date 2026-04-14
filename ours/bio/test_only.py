import torch
import torch.utils.data as Data
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from dataset import *
from model import *

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / (mcc_den + 1e-8)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        roc_auc = 0.5
        pr_auc = 0.5
    return {
        'Sn': sensitivity,
        'Sp': specificity,
        'Acc': accuracy,
        'MCC': mcc,
        'Precision': precision,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }

# ── 加载独立测试集 ──────────────────────────────────────
sequences, labels = load_encoding_from_txt('data-200-shuffled.fasta')
features = load_features_from_txt('features_200.txt')

sequences = np.array(sequences)
features  = np.array(features)
labels    = np.array(labels)

print(f"测试集大小: {len(labels)}, 正类: {int(labels.sum())}, 负类: {int((1-labels).sum())}")
print(f"特征均值前5: {np.mean(features, axis=0)[:5]}")
print(f"特征标准差前5: {np.std(features, axis=0)[:5]}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MyDataSet(sequences, features, labels)
loader  = Data.DataLoader(dataset, batch_size=64, shuffle=False)

# ── 加载单个权重 ────────────────────────────────────────
model = FusionPepNet().to(device)
model.load_state_dict(torch.load("fold_1.pth", map_location=device))
model.eval()

all_probs  = []
all_preds  = []
all_labels = []

with torch.no_grad():
    for input_ids, sequence_features, lbls in loader:
        input_ids         = input_ids.to(device)
        sequence_features = sequence_features.to(device)
        outputs, _, _     = model(input_ids, sequence_features)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# 诊断信息
print(f"\nprob 均值: {all_probs.mean():.4f}")
print(f"prob 最大: {all_probs.max():.4f}")
print(f"prob 最小: {all_probs.min():.4f}")
print(f"预测为正类的数量: {all_preds.sum()}")
print(f"真实正类的数量:   {all_labels.sum()}")

metrics = calculate_metrics(all_labels, all_preds, all_probs)
print("\n====== 独立测试集结果 ======")
for k, v in metrics.items():
    print(f"  {k:12}: {v:.4f}")