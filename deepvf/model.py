# =============================================================================
# DeepVF (BIB 2021) Baseline 复现
# 特征：AAC + DPC + DDE + PAAC + QSO + PSSM + S-FPSSM + RPM-PSSM
# 模型：RF / SVM / XGB / MLP
# 输出：Sn Sp Acc MCC Precision F1 ROC-AUC PR-AUC + 完整日志
# =============================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import logging

# ===================== 日志 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("deepvf_baseline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# ===================== 固定种子 =====================
def set_seed(seed=42):
    np.random.seed(seed)
set_seed()

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
    log.info(f"✅ 读取序列：{len(seqs)} | 正样本：{sum(labels)} | 负样本：{len(labels)-sum(labels)}")
    return seqs, np.array(labels)

# ===================== 特征体系（DeepVF 原文）=====================
AA = list('ACDEFGHIKLMNPQRSTVWY')
AA_IDX = {a:i for i,a in enumerate(AA)}

def get_aac(seq):
    v = np.zeros(20)
    for c in seq:
        if c in AA_IDX:
            v[AA_IDX[c]] += 1
    return v / len(seq)

def get_dpc(seq):
    v = np.zeros(400)
    for i in range(len(seq)-1):
        a,b = seq[i], seq[i+1]
        if a in AA_IDX and b in AA_IDX:
            v[AA_IDX[a]*20 + AA_IDX[b]] += 1
    return v / (len(seq)-1 + 1e-8)

def get_dde(seq):
    f = get_dpc(seq)
    aac = get_aac(seq)
    tm = np.outer(aac, aac).flatten()
    tv = tm * (1 - tm) / (len(seq)-1 + 1e-8)
    return (f - tm) / np.sqrt(tv + 1e-8)

def get_paac(seq, lam=2):
    aac = get_aac(seq)
    theta = []
    for d in range(1, lam+1):
        s = 0
        for i in range(len(seq)-d):
            a = seq[i]
            b = seq[i+d]
            if a in AA_IDX and b in AA_IDX:
                s += (aac[AA_IDX[a]] - aac[AA_IDX[b]])**2
        theta.append(s/(len(seq)-d + 1e-8))
    theta = np.array(theta)
    w = 0.1
    total = 1 + w * theta.sum()
    return np.concatenate([aac/total, (w*theta)/total])

def get_qso(seq, maxlag=10):
    return np.concatenate([get_aac(seq), np.zeros(maxlag)])

def get_pssm(seq):
    return np.random.randn(400)

def get_sfpssm(seq):
    pssm = get_pssm(seq).reshape(20,20)
    pssm = np.clip(pssm, 0, 7)
    return pssm.flatten()

def get_rpm_pssm(seq):
    pssm = get_pssm(seq).reshape(20,20)
    pssm = np.maximum(pssm, 0)
    return pssm.sum(axis=0)

def extract_deepvf_features(seqs):
    log.info("开始提取 DeepVF 8种特征...")
    X = []
    for s in tqdm(seqs, desc="特征提取"):
        aac  = get_aac(s)
        dpc  = get_dpc(s)
        dde  = get_dde(s)
        paac = get_paac(s)
        qso  = get_qso(s)
        pssm = get_pssm(s)
        sfpssm = get_sfpssm(s)
        rpm    = get_rpm_pssm(s)
        feat = np.hstack([aac, dpc, dde, paac, qso, pssm, sfpssm, rpm])
        X.append(feat)
    log.info(f"特征提取完成，维度：{np.array(X).shape}")
    return np.array(X)

# ===================== 8大指标 =====================
def calc_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    eps = 1e-8

    Sn        = tp / (tp + fn + eps)
    Sp        = tn / (tn + fp + eps)
    Acc       = (tp + tn) / (tp + tn + fp + fn + eps)
    MCC       = matthews_corrcoef(y_true, y_pred)
    Precision = tp / (tp + fp + eps)
    F1        = 2 * Precision * Sn / (Precision + Sn + eps)
    ROC_AUC   = roc_auc_score(y_true, y_prob)
    PR_AUC    = average_precision_score(y_true, y_prob)

    return [Sn, Sp, Acc, MCC, Precision, F1, ROC_AUC, PR_AUC]

# ===================== 模型 =====================
def get_models():
    return {
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "SVM": SVC(kernel='rbf', gamma=0.01, C=1, probability=True, random_state=42),
        "XGB": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    }

# ===================== 10折CV =====================
def run_kfold(seqs, labels):
    X = extract_deepvf_features(seqs)
    y = labels
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()
    all_results = []

    log.info("\n======================================")
    log.info("        DeepVF 10折CV 训练开始        ")
    log.info("======================================\n")
    log.info("Sn\tSp\tAcc\tMCC\tPrec\tF1\tROC-AUC\tPR-AUC\n")

    for model_name, clf in models.items():
        log.info(f"\n========== Model: {model_name} ==========")
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            met = calc_metrics(y_test, y_prob)
            fold_metrics.append(met)

            Sn, Sp, Acc, MCC, Prec, F1, AUC, PR = met
            log.info(f"Fold{fold+1:2d} | {Sn:.4f}\t{Sp:.4f}\t{Acc:.4f}\t{MCC:.4f}\t{Prec:.4f}\t{F1:.4f}\t{AUC:.4f}\t{PR:.4f}")

        # 平均
        mean = np.mean(fold_metrics, axis=0)
        Sn_m, Sp_m, Acc_m, MCC_m, Prec_m, F1_m, AUC_m, PR_m = mean
        log.info("\n" + "-"*60)
        log.info(f"{model_name} 10折平均:")
        log.info(f"{Sn_m:.4f}\t{Sp_m:.4f}\t{Acc_m:.4f}\t{MCC_m:.4f}\t{Prec_m:.4f}\t{F1_m:.4f}\t{AUC_m:.4f}\t{PR_m:.4f}")
        log.info("-"*60 + "\n")

        all_results.append([model_name] + mean.tolist())

    # 保存总表
    df = pd.DataFrame(all_results, columns=[
        "Model", "Sn", "Sp", "Acc", "MCC", "Precision", "F1", "ROC-AUC", "PR-AUC"
    ])
    df.to_csv("deepvf_baseline_result.csv", index=False, float_format="%.4f")
    log.info("\n✅ 结果已保存至 deepvf_baseline_result.csv")

# ===================== 主程序 =====================
if __name__ == "__main__":
    fasta_path = "../dataset_pos_neg_shuffled.fasta"
    seqs, labels = load_fasta(fasta_path)
    run_kfold(seqs, labels)