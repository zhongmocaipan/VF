from Bio import SeqIO
import sys

def rename_fasta(input_file, output_file, prefix):
    records = []
    for idx, rec in enumerate(SeqIO.parse(input_file, "fasta"), 1):
        rec.id = f"{prefix}_{idx}"       # 新名字：pos_1, neg_2...
        rec.description = ""            # 清空多余描述
        records.append(rec)
    
    SeqIO.write(records, output_file, "fasta")
    print(f"✅ 完成：{output_file}")

# ===================== 运行 =====================
rename_fasta("pos_100.fasta", "pos_100_renamed.fasta", "pos")
rename_fasta("neg_100.fasta", "neg_100_renamed.fasta", "neg")

print("\n🎉 所有序列名称已修改完成！")