# import esm
# import torch
# import numpy as np
# from typing import List
# from tqdm import tqdm
# import gc
# import os

# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# torch.backends.cudnn.enabled = False

# MAX_SEQ_LEN = 1024

# def read_protein_sequences_from_fasta(fasta_path: str) -> List[str]:
#     sequences = []
#     current_seq = ""
#     with open(fasta_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith('>'):
#                 if current_seq:
#                     current_seq = current_seq[:MAX_SEQ_LEN]
#                     sequences.append(current_seq)
#                     current_seq = ""
#             else:
#                 current_seq += line
#         if current_seq:
#             current_seq = current_seq[:MAX_SEQ_LEN]
#             sequences.append(current_seq)
#     return sequences


# class ESMFeatureExtractor:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"✅ 运行设备: {self.device}")
#         self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#         self.esm_model.eval()
#         self.esm_model.to(self.device)
#         self.batch_converter = self.alphabet.get_batch_converter()

#     def extract_features(self, sequences: List[str]) -> np.ndarray:
#         all_features = []
#         with tqdm(total=len(sequences), desc="生成ESM特征") as pbar:
#             for seq in sequences:
#                 batch_labels, batch_strs, batch_tokens = self.batch_converter([("0", seq)])
#                 batch_tokens = batch_tokens.to(self.device)
#                 attention_mask = (batch_tokens != self.alphabet.padding_idx).bool()

#                 with torch.no_grad():
#                     results = self.esm_model(
#                         batch_tokens,
#                         repr_layers=[self.esm_model.num_layers],
#                         return_contacts=False
#                     )

#                 rep = results["representations"][self.esm_model.num_layers]  # [1, L, D]
#                 mask = attention_mask.unsqueeze(-1).expand_as(rep).float()
#                 feat = (rep * mask).sum(dim=1) / attention_mask.float().sum(dim=1, keepdim=True)

#                 all_features.append(feat.detach().cpu().numpy())

#                 del results, rep, feat, batch_tokens, attention_mask, mask
#                 torch.cuda.empty_cache()
#                 gc.collect()
#                 pbar.update(1)

#         return np.concatenate(all_features, axis=0)


# if __name__ == "__main__":
#     FASTA_PATH  = r"data-200-shuffled.fasta"
#     OUTPUT_PATH = r"features_200.txt"

#     print("读取fasta序列...")
#     sequences = read_protein_sequences_from_fasta(FASTA_PATH)
#     print(f"共读取 {len(sequences)} 条序列")

#     extractor = ESMFeatureExtractor()
#     features = extractor.extract_features(sequences)

#     np.savetxt(OUTPUT_PATH, features, fmt="%.6f")
#     print(f"✅ 完成！shape = {features.shape}")
import esm
import torch
import numpy as np
from typing import List
from tqdm import tqdm
import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.enabled = False

MAX_SEQ_LEN = 1024

def read_protein_sequences_from_fasta(fasta_path: str) -> List[str]:
    sequences = []
    current_seq = ""
    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    current_seq = current_seq[:MAX_SEQ_LEN]
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line
        if current_seq:
            current_seq = current_seq[:MAX_SEQ_LEN]
            sequences.append(current_seq)
    return sequences


class ESMFeatureExtractor:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ 运行设备: {self.device}")

        # 根据model_name选择模型
        if model_name == "esm2_t33_650M_UR50D":
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model_name == "esm2_t12_35M_UR50D":
            self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif model_name == "esm2_t6_8M_UR50D":
            self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        print(f"✅ 使用模型: {model_name}, embed_dim: {self.esm_model.embed_dim}")
        self.esm_model.eval()
        self.esm_model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    def extract_features(self, sequences: List[str]) -> np.ndarray:
        all_features = []
        with tqdm(total=len(sequences), desc="生成ESM特征") as pbar:
            for seq in sequences:
                batch_labels, batch_strs, batch_tokens = self.batch_converter([("0", seq)])
                batch_tokens = batch_tokens.to(self.device)
                attention_mask = (batch_tokens != self.alphabet.padding_idx).bool()

                with torch.no_grad():
                    results = self.esm_model(
                        batch_tokens,
                        repr_layers=[self.esm_model.num_layers],
                        return_contacts=False
                    )

                rep = results["representations"][self.esm_model.num_layers]  # [1, L, D]
                mask = attention_mask.unsqueeze(-1).expand_as(rep).float()
                feat = (rep * mask).sum(dim=1) / attention_mask.float().sum(dim=1, keepdim=True)

                all_features.append(feat.detach().cpu().numpy())

                del results, rep, feat, batch_tokens, attention_mask, mask
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)

        return np.concatenate(all_features, axis=0)


if __name__ == "__main__":
    FASTA_PATH  = r"data-200-shuffled.fasta"
    OUTPUT_PATH = r"features_200.txt"

    # ── 在这里选择模型 ──────────────────────────────────────
    # esm2_t6_8M_UR50D   → embed_dim=320  (最小，最快)
    # esm2_t12_35M_UR50D → embed_dim=480  (中等)
    # esm2_t33_650M_UR50D→ embed_dim=1280 (最大，效果最好，需要较多显存)
    MODEL_NAME = "esm2_t33_650M_UR50D"

    print("读取fasta序列...")
    sequences = read_protein_sequences_from_fasta(FASTA_PATH)
    print(f"共读取 {len(sequences)} 条序列")

    extractor = ESMFeatureExtractor(model_name=MODEL_NAME)
    features = extractor.extract_features(sequences)

    np.savetxt(OUTPUT_PATH, features, fmt="%.6f")
    print(f"✅ 完成！shape = {features.shape}")