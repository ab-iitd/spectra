import re
from typing import Optional
from collections import Counter

# REGEX patterns
pattern2 = re.compile(r"""
(
    <think_reasoning>.*?</think_reasoning>\s*
    (?:[^<]*?)?
    <tool_call>.*?</tool_call>\s*
    (?:[^<]*?)?
    <tool_response>.*?</tool_response>\s*
    (?:[^<]*?)?
    <think_perception>.*?</think_perception>\s*
    (?:[^<]*?)?
    <think_reasoning>.*?</think_reasoning>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL | re.IGNORECASE)

pattern4 = re.compile(r"""
(
    <think_reasoning>.*?</think_reasoning>\s*
    (?:[^<]*?)?
    <think_perception>.*?</think_perception>\s*
    (?:[^<]*?)?
    <think_reasoning>.*?</think_reasoning>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL | re.IGNORECASE)

pattern5 = re.compile(r"""
(
    <think_reasoning>.*?</think_reasoning>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL | re.IGNORECASE)

# Normalization Helpers
def normalize_math_expr(ans: str) -> str:
    """Convert LaTeX math expressions to simple strings for comparison."""
    ans = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt\1", ans)
    ans = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", ans)
    ans = ans.replace("{", "").replace("}", "")
    return ans.strip()

def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.strip()
    boxed_match = re.match(r"\\boxed\{(.*)\}", final_answer)
    if boxed_match:
        final_answer = boxed_match.group(1).strip()

    final_answer = normalize_math_expr(final_answer)

    paren_match = re.match(r"\(?\s*([A-Z])\s*\)?", final_answer)
    if paren_match:
        return paren_match.group(1)

    match = re.search(r"[A-Z]", final_answer)
    if match:
        return match.group(0)

    return final_answer.strip()

# Reward Components
def correct_answer_reward(solution_str: str, ground_truth: str) -> tuple[bool, str, int]:
    ans_tags = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    num_answers = len(ans_tags)

    pred = ""
    if num_answers > 0:
        pred = normalize_final_answer(ans_tags[-1])

    last_part = solution_str[-200:]  # slightly larger heuristic slice

    boxed_match = re.search(r"\\boxed\{([^}]*)\}", last_part)
    if boxed_match:
        boxed_candidate = normalize_final_answer(boxed_match.group(1))
        if boxed_candidate.strip():
            pred = boxed_candidate
    else:
        broken_boxed = re.search(r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?", last_part)
        if broken_boxed:
            pred = broken_boxed.group(1)

    pred_clean = re.sub(r"\s+", "", pred).upper()
    gt_clean = re.sub(r"\s+", "", ground_truth).upper()

    is_correct = (pred_clean == gt_clean)
    return is_correct, pred, num_answers


#1.1. FORMAT REWARD 
def format_reward(predict_str: str) -> float: # ------------------------------> 2.0 
    rw_f = 2.0
    lp = 0.75
    if pattern2.search(predict_str):
        return rw_f
    elif pattern4.search(predict_str):
        return rw_f*lp
    elif pattern5.search(predict_str):
        return rw_f*lp*lp
    return 0.0

# 1.2. STRICT FORMAT REWARD — answer at the end, only one <answer> tag
def strict_format_reward(predict_str: str) -> float:    # ------------------------------> 3.0 
    answer_at_end = re.search(r"<answer>.*?</answer>\s*$", predict_str, re.DOTALL)
    ans_tags = re.findall(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)

    if answer_at_end and len(ans_tags) == 1:
        return 2.0
    if len(ans_tags) > 0:
        return 2.0
    else:
        return -1.0

# Tool Reward: 
def correct_tool_format_reward(predict_str: str) -> float:
    """Reward for correct tool call format structure"""
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)
    tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", predict_str, re.DOTALL)
    reward = 0.0

    # Tool call and response count match
    if len(tool_calls) == len(tool_responses) and len(tool_calls) > 0:
        reward += 0.5
    elif len(tool_calls) != len(tool_responses):
        reward -= 0.25

    # Validate tool_call content structure
    for call in tool_calls:
        name = re.search(r'"name"\s*:\s*"(python_tool|ocr_tool|detection_tool|captioning_tool|perception_tool|search_tool)"', call)
        arguments = re.search(r'"arguments"\s*:\s*(\\?\{.*?\\?\}|"(?:\\.|[^"])*")', call, re.DOTALL)

        if name and arguments:
            reward += 0.1
        elif name:
            reward += 0.05

    return min(reward, 0.8)    # -----------------0.8

def tool_usage_reward(predict_str: str) -> float:
    """Reward for actually using tools (any tool usage is good)"""
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)

    if len(tool_calls) > 0:
        return 1.0
    return 0.0

def tool_success_reward(predict_str: str) -> float:
    """Reward for successful tool execution"""
    pattern_success = re.compile(r"<tool_response>.*?\{\"success\":\s*true.*?\}.*?</tool_response>", re.DOTALL)
    
    if pattern_success.search(predict_str):
        return 1.0
    return -0.5

def combined_tool_reward(predict_str: str) -> float:
    """Combined reward for all tool-related behaviors"""
    format_reward = correct_tool_format_reward(predict_str)
    usage_reward = tool_usage_reward(predict_str)
    success_reward = tool_success_reward(predict_str)
    
    return format_reward + usage_reward + success_reward # ------------------> 2.8

# Final Score Computation
def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    format_score: float = 0.2
):
    
    correct, pred, num_answers = correct_answer_reward(solution_str, ground_truth)

    if correct:
        acc_reward = 8.0 / 1.0
    else:
        acc_reward = -2.0

    abs_max = 8 + 2 + (3 * 2.0) + (2 * 2.8)  # Ans rew + format_rew + strict_ans_rew + tool_reward... Max possible score components
    
    format_rew = format_reward(solution_str)
    
    strict_ans_rew = strict_format_reward(solution_str)
    tool_rew = combined_tool_reward(solution_str)


    final_score = 2.5 * ( 1.0 * acc_reward + 1.0 * format_rew + 3.0 * strict_ans_rew + 2.0 * tool_rew) / abs_max

    return {
        "score": final_score,
        "acc_reward": 2.5 * acc_reward / abs_max,
        "format_reward": 2.5 * format_rew / abs_max,
        "strict_answer_reward": 2.5 * strict_ans_rew / abs_max,
        "tool_reward": 2.5 * tool_rew / abs_max,
        "acc": correct,
        "pred": pred,
    }