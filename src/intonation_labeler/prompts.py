from __future__ import annotations

import json
from typing import Any


def build_prompt(transcript: str, anchors: dict[str, Any]) -> str:
    return f"""
# 任务
你是一位专业的语音标注员。请结合【音频听感】与提供的【声学参数数据】，在文本的关键位点标注语调。

# 原始文本
"{transcript}"

# 声学参数定义
- t_maxf0: 最高基频出现时间点
- z_f0: 标准化音高分数
- rangeF0: 音域跨度
- slope: 基频曲线斜率（负号表示下降趋势）
- t_min: 最低基频出现时间点

# 待分析位点数据（仅分析以下 tag）
{json.dumps(anchors, ensure_ascii=False, indent=2)}

# 标注要求
1. **优先末尾走向（严禁机械看斜率）**：
   - 语调判断应以单词**末尾音节的实际走向**为准。
   - 即使 `slope` 为负数，如听感末尾有上扬或趋于平稳，应标【上升】或【平调】。
2. **严禁语义干扰**：标注必须严格基于数值和听感，不受句型影响。
3. **位置强制要求**：语调标签必须紧跟在对应单词及其标点符号的【后面】。
4. **文本原貌约束**：严禁改动原始文本，不得增删空格、标点，必须原样保留 '#' 停顿标识。
5. **每个 anchor tag 均须输出标签**：必须选择【上升】【下降】【平调】之一，不得省略任何位点。
6. **输出格式**：仅输出标注后的文本，不解释。示例：Hi,【下降】 how are you?【上升】

请开始标注：
"""

