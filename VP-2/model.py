# =============================================================================
# VirulentPred 2.0 最终完美版 —— 每一折都输出全部 8 个指标 + LOG日志
# 输出：Sn  Sp  Acc  MCC  Precision  F1  ROC-AUC  PR-AUC
# =============================================================================
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

# ===================== 日志模块 =====================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("train_virulentpred2.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# ===================== FASTA 读取 =====================
def load_fasta(path):
    seqs, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    seq = ''
    for line in lines:
        if line.startswith('>'):
            if seq:
                seqs.append(seq)
                seq = ''
            labels.append(1 if 'pos' in line else 0)
        else:
            seq += line
    if seq:
        seqs.append(seq)
    log.info(f"✅ 读取序列：{len(seqs)} | 正：{sum(labels)} | 负：{len(labels)-sum(labels)}")
    return seqs, np.array(labels)

# ===================== AAC + DPC 特征 =====================
AA = 'ACDEFGHIKLMNPQRSTVWY'

def aac(seq):
    cnt = {c:0 for c in AA}
    for c in seq:
        if c in cnt:
            cnt[c] += 1
    total = len(seq) or 1
    return [cnt[c]/total for c in AA]

def dpc(seq):
    pairs = [a+b for a in AA for b in AA]
    cnt = {p:0 for p in pairs}
    for i in range(len(seq)-1):
        p = seq[i:i+2]
        if p in cnt:
            cnt[p] += 1
    total = len(seq)-1 or 1
    return [cnt[p]/total for p in pairs]

def extract_features(seqs):
    X = []
    for s in tqdm(seqs, desc="提取特征"):
        X.append(aac(s) + dpc(s))
    return np.array(X)

# ===================== 8个指标 =====================
def calculate_metrics(y_true, y_pred, y_prob):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    Sn = tp / (tp + fn + 1e-8)
    Sp = tn / (tn + fp + 1e-8)
    Acc = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    Precision = precision_score(y_true, y_pred, zero_division=0)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    ROC_AUC = roc_auc_score(y_true, y_prob)
    PR_AUC = average_precision_score(y_true, y_prob)

    return [Sn, Sp, Acc, MCC, Precision, F1, ROC_AUC, PR_AUC]

# ===================== 10折交叉验证 =====================
def run_kfold(seqs, labels):
    X = extract_features(seqs)
    y = labels
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    # 打印表头（每折都显示）
    log.info("\nSn\tSp\tAcc\tMCC\tPrec\tF1\tROC-AUC\tPR-AUC")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        log.info(f"\n========== Fold {fold+1}/10 ==========")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        results.append(metrics)

        # ========== 这里修复：每一折都输出全部8个指标 ==========
        Sn, Sp, Acc, MCC, Precision, F1, ROC_AUC, PR_AUC = metrics
        log.info(f"{Sn:.4f}\t{Sp:.4f}\t{Acc:.4f}\t{MCC:.4f}\t{Precision:.4f}\t{F1:.4f}\t{ROC_AUC:.4f}\t{PR_AUC:.4f}")

    # 输出最终平均
    mean = np.mean(results, axis=0)
    log.info("\n" + "="*70)
    log.info(" VirulentPred 2.0 10折平均结果 ")
    log.info("="*70)
    log.info("Sn\tSp\tAcc\tMCC\tPrec\tF1\tROC-AUC\tPR-AUC")
    log.info(f"{mean[0]:.4f}\t{mean[1]:.4f}\t{mean[2]:.4f}\t{mean[3]:.4f}\t{mean[4]:.4f}\t{mean[5]:.4f}\t{mean[6]:.4f}\t{mean[7]:.4f}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    path = "../dataset_pos_neg_shuffled.fasta"
    seqs, labels = load_fasta(path)
    run_kfold(seqs, labels)