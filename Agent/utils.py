import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class SectionPlan:
    id: int
    title: str
    focus: str
    must_cover: List[str]
    retrieval_hints: List[str]


class PlannerParseError(Exception):
    pass


def _extract_answer_block(text: str) -> str:
    """提取 <answer>...</answer> 之间的内容；若没有则抛错。"""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if not m:
        raise PlannerParseError("未找到 <answer> 块")
    return m.group(1).strip()


def _strip_code_fences(s: str) -> str:
    """去掉可能出现的 ```json ... ``` 或 ``` ... ``` 包裹。"""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _basic_json_sanitize(s: str) -> str:
    """对常见小问题做轻量修复（尽量保守）：去 BOM、全角空格、尾随逗号等。"""
    s = s.replace("\ufeff", "")                # BOM
    s = s.replace("\u3000", " ")               # 全角空格
    # 去掉对象/数组里的尾随逗号（非常保守的匹配）
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_planner_response(text: str) -> Dict[str, Any]:
    """
    解析 LLM 的 planner 响应，返回字典：
    {
      "sections": [ {id, title, focus, must_cover, retrieval_hints}, ... ]
    }
    若解析失败抛 PlannerParseError。
    """
    # 1) 抽取 <answer> JSON
    answer_raw = _extract_answer_block(text)
    answer_raw = _strip_code_fences(answer_raw)
    answer_raw = _basic_json_sanitize(answer_raw)

    # 2) JSON 解析
    try:
        payload = json.loads(answer_raw)
    except json.JSONDecodeError as e:
        # 尝试再缩一遍空白并提示错误上下文
        snippet = answer_raw[max(0, e.pos-80): e.pos+80]
        raise PlannerParseError(f"JSON 解析失败: {e}; 片段: {snippet!r}")

    # 3) 结构校验
    if not isinstance(payload, dict) or "sections" not in payload:
        raise PlannerParseError("JSON 中缺少 'sections' 字段")

    sections = payload["sections"]
    if not isinstance(sections, list) or not (2 <= len(sections) <= 4):
        raise PlannerParseError(f"'sections' 数量应为 2–4，当前 {len(sections)}")

    # 4) 字段与 id 连续性校验，并转为 SectionPlan
    plans: List[SectionPlan] = []
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            raise PlannerParseError(f"sections[{i}] 不是对象")
        try:
            sid = int(sec["id"])
            title = str(sec["title"]).strip()
            focus = str(sec["focus"]).strip()
            must_cover = list(sec["must_cover"])
            retrieval_hints = list(sec["retrieval_hints"])
        except Exception as e:
            raise PlannerParseError(f"sections[{i}] 字段缺失或类型错误: {e}")

        if sid != i:
            raise PlannerParseError(f"sections[{i}].id 应为 {i}，实际为 {sid}")
        if not (3 <= len(must_cover) <= 6):
            raise PlannerParseError(f"sections[{i}].must_cover 数量应为 3–6")
        if not (5 <= len(retrieval_hints) <= 10):
            raise PlannerParseError(f"sections[{i}].retrieval_hints 数量应为 5–10")

        plans.append(SectionPlan(
            id=sid, title=title, focus=focus,
            must_cover=[str(x).strip() for x in must_cover if str(x).strip()],
            retrieval_hints=[str(x).strip() for x in retrieval_hints if str(x).strip()],
        ))

    # 5) 返回规范化结果
    return {"sections": [p.__dict__ for p in plans]}
