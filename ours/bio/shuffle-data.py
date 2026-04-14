import random

def shuffle_fasta(input_fa, output_fa):
    # 读取 FASTA
    with open(input_fa, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]

    records = []
    current_header = None
    current_seq = []

    for line in lines:
        if line.startswith('>'):
            if current_header is not None:
                records.append((current_header, ''.join(current_seq)))
            current_header = line
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_header is not None:
        records.append((current_header, ''.join(current_seq)))

    # 打乱
    random.seed(42)
    random.shuffle(records)

    # 写入新文件
    with open(output_fa, 'w', encoding='utf-8') as f:
        for h, s in records:
            f.write(f"{h}\n")
            f.write(f"{s}\n")

    print(f"✅ 打乱完成！已保存到: {output_fa}")
    print(f"总序列数: {len(records)}")


# ===================== 在这里改你的文件名 =====================
if __name__ == "__main__":
    shuffle_fasta("data-200.fasta", "data-200-shuffled.fasta")