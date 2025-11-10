#!/usr/bin/env python3
"""临时补丁：修复维度不匹配问题"""

import shutil
import os

# 备份原文件
src = "recbole/model/sequential_recommender/sasrec_align.py"
backup = "recbole/model/sequential_recommender/sasrec_align.py.bak"

if os.path.exists(src) and not os.path.exists(backup):
    shutil.copy(src, backup)
    print(f"已备份: {backup}")

# 读取文件
with open(src, 'r') as f:
    content = f.read()

# 替换检查逻辑
old_check = """        if emb.size(0) != expected_rows:
            return None"""

new_check = """        if emb.size(0) != expected_rows:
            self.logger.warning(
                f"Text embedding size mismatch: got {emb.size(0)}, expected {expected_rows}. "
                f"Will pad/truncate to match."
            )
            # 调整大小以匹配
            if emb.size(0) < expected_rows:
                # 填充零
                pad_rows = expected_rows - emb.size(0)
                pad = torch.zeros((pad_rows, emb.size(1)), dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            else:
                # 截断
                emb = emb[:expected_rows]"""

if old_check in content:
    content = content.replace(old_check, new_check)
    # 添加torch导入
    if "import torch" not in content:
        content = content.replace("from torch import nn", "from torch import nn\nimport torch")
    
    with open(src, 'w') as f:
        f.write(content)
    print("✓ 已应用补丁")
else:
    print("⚠️  未找到需要修改的代码")
