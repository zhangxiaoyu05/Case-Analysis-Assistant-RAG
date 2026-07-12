"""
将 20种药品说明书合集.txt 拆分为独立的 20 个 txt 文件。

使用方式:
    python scripts/split_drug_file.py

输出:
    data/raw/<通用名称>.txt × 20 个文件
"""

import re
import sys
from pathlib import Path

# Windows 下强制 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "20种药品说明书合集.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# 读取合集文件
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# 按分隔符拆分（匹配一行全等号的行）
sections = re.split(r"\n?^=+\s*$", full_text, flags=re.MULTILINE)
sections = [s.strip() for s in sections if s.strip()]

print(f"共检测到 {len(sections)} 个药品段落\n")

created_files = []

for section in sections:
    lines = section.strip().split("\n")

    # 提取通用名称
    generic_name = None
    for line in lines[:10]:
        m = re.match(r"通用名称[：:]\s*(.+)", line.strip())
        if m:
            generic_name = m.group(1).strip()
            break

    if not generic_name:
        # 回退：用第一行去掉"说明书"后缀
        first_line = lines[0].strip() if lines else "unknown"
        generic_name = re.sub(r"说明书$", "", first_line)
        print(f"[WARN] 未找到通用名称，使用标题: {generic_name}")

    # 生成文件名
    filename = generic_name.replace("/", "-").replace("\\", "-").replace(":", "：")
    filepath = OUTPUT_DIR / f"{filename}.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(section + "\n")

    created_files.append(filepath)
    print(f"[OK] {filepath.name}")

print(f"\n拆分完成！共 {len(created_files)} 个文件，保存在 {OUTPUT_DIR}")
