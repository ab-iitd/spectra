import re
from typing import Optional
from collections import Counter

# -------------------------------------------------------
# REGEX patterns
# -------------------------------------------------------
# pattern2 — reasoning + one tool bundle + perception + reasoning + answer
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

# pattern4 — reasoning → perception → reasoning → answer (no tool)
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

# pattern5 — reasoning → answer (minimal)
pattern5 = re.compile(r"""
(
    <think_reasoning>.*?</think_reasoning>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL | re.IGNORECASE)

# -------------------------------------------------------
# Normalization Helpers
# -------------------------------------------------------
def normalize_math_expr(ans: str) -> str:
    """Convert LaTeX math expressions to simple strings for comparison."""
    # sqrt{...} → sqrt...
    ans = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt\1", ans)

    # \frac{a}{b} → a/b
    ans = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", ans)

    # Remove extra braces
    ans = ans.replace("{", "").replace("}", "")

    return ans.strip()

def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.strip()

    # Remove \boxed{...} if fully closed
    boxed_match = re.match(r"\\boxed\{(.*)\}", final_answer)
    if boxed_match:
        final_answer = boxed_match.group(1).strip()

    # Normalize math
    final_answer = normalize_math_expr(final_answer)

    # Pattern: (A) or ( B ) etc.
    paren_match = re.match(r"\(?\s*([A-Z])\s*\)?", final_answer)
    if paren_match:
        return paren_match.group(1)

    # Fallback: first uppercase letter
    match = re.search(r"[A-Z]", final_answer)
    if match:
        return match.group(0)

    return final_answer.strip()

# -------------------------------------------------------
# Reward Components
# -------------------------------------------------------
def correct_answer_reward(solution_str: str, ground_truth: str) -> tuple[bool, str, int]:
    ans_tags = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    num_answers = len(ans_tags)

    pred = ""
    if num_answers > 0:
        pred = normalize_final_answer(ans_tags[-1])

    # ---- NEW LOGIC: Look for boxed answers ----
    last_part = solution_str[-200:]  # slightly larger heuristic slice

    # Case 1: Normal boxed { } form
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", last_part)
    if boxed_match:
        boxed_candidate = normalize_final_answer(boxed_match.group(1))
        if boxed_candidate.strip():
            pred = boxed_candidate
    else:
        # Case 2: Missing closing brace -> \boxed{(A)
        broken_boxed = re.search(r"\\boxed\{\s*\(?\s*([A-Z])\s*\)?", last_part)
        if broken_boxed:
            pred = broken_boxed.group(1)

    # Normalize and compare
    pred_clean = re.sub(r"\s+", "", pred).upper()
    gt_clean = re.sub(r"\s+", "", ground_truth).upper()

    is_correct = (pred_clean == gt_clean)
    return is_correct, pred, num_answers

###################################################################################################
#2. FORMAT REWARD WITH LENGTH PENALTY
def format_reward(predict_str: str) -> float:
    rw_f = 2.0
    lp = 0.75
    if pattern2.search(predict_str):
        return rw_f
    elif pattern4.search(predict_str):
        return rw_f * lp
    elif pattern5.search(predict_str):
        return rw_f * lp * 0.75
    return 0.0


# 3. TOOL CALL FORMAT REWARD
def correct_tool_format_reward(predict_str: str) -> float:
    # Extract tool_call and tool_response contents
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)
    tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", predict_str, re.DOTALL)
    reward = 0.0

    # Tool call and response count match---------------------------------------------------------------------------------------
    if len(tool_calls) == len(tool_responses):
        reward += 0.5
    else:
        reward -= 0.25

    # Tool call encouraged ---------------------------------------------------------------------------------------------------
    allowed_tools = {"python_tool", "ocr_tool", "detection_tool", "captioning_tool", "perception_tool", "search_tool"}
    tool_usage = Counter()

    for call in tool_calls:
        match = re.search(r'"name"\s*:\s*"([^"]+)"', call)
        if match:
            tool_name = match.group(1)
            if tool_name in allowed_tools:
                tool_usage[tool_name] += 1

    #To encourage tool calls 
    if all(1 < count <= 2 for count in tool_usage.values()):  # old 0.5
        reward += 0.5
    else:
        reward -= 0.25


    # --- Validate tool_call content structure ---
    for call in tool_calls:
        name = re.search(r'"name"\s*:\s*"(python_tool|ocr_tool|detection_tool|captioning_tool|perception_tool|search_tool)"',call)
        arguments = re.search(r'"arguments"\s*:\s*(\\?\{.*?\\?\}|"(?:\\.|[^"])*")', call, re.DOTALL)

        # Reward structure quality
        if name and arguments:
            reward += 0.1
        elif name:
            reward += 0.05

    return min(reward, 1.5)  # old 2.1


# 4. STRICT FORMAT REWARD — answer at the end, only one <answer> tag
def strict_format_reward(predict_str: str) -> float:
    answer_at_end = re.search(r"<answer>.*?</answer>\s*$", predict_str, re.DOTALL)
    ans_tags = re.findall(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)

    if answer_at_end and len(ans_tags) == 1:
        return 8.0
    if len(ans_tags) > 0:
        return 8.0
    else:
        return -5.0
    
###### TO fix issue of n number of \n
def space_format_reward(predict_str: str) -> float:
    # Detect long newline sequences (4 or more)
    if re.search(r"(?:\n\s*){6,}", predict_str):
        return -25.0
    return 0.0

# 5. TOOL CALL rool_response back to back reward
def tool_call_presence_reward(predict_str: str) -> float:
    # Pattern to detect a complete tool call + response block
    pattern_bundle = re.compile(r"<tool_call>.*?</tool_call>user\n<tool_response>.*?</tool_response>",re.DOTALL)
    pattern_bundle_2 = re.compile(r"<tool_call>.*?</tool_call>\nuser\n<tool_response>.*?</tool_response>",re.DOTALL)
    # Pattern to detect success: true inside tool_response
    pattern_success = re.compile(r"<tool_response>.*?\{\"success\":\s*true.*?\}.*?</tool_response>",re.DOTALL)

    reward = 0.0

    # Check bundle presence
    if pattern_bundle.search(predict_str) or pattern_bundle_2.search(predict_str):
        reward += 1.0
    else:
        reward -= 0.5
        
    # Check success presence
    if pattern_success.search(predict_str):
        reward += 0.5
    else:
        reward -= 0.25

    return reward

def empty_tool_format_reward(predict_str: str) -> float:
    reward = 0.0

    # Extract tool_call blocks
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)

    for call in tool_calls:
        # Match only search_tool
        if re.search(r'"name"\s*:\s*"search_tool"', call):

            # Capture the arguments object (even with spaces)
            arg_match = re.search(
                r'"arguments"\s*:\s*(\{.*?\})',
                call,
                re.DOTALL
            )

            if arg_match:
                arguments_content = arg_match.group(1).strip()

                # Check if arguments are empty like {} or {   }
                if re.fullmatch(r"\{\s*\}", arguments_content):
                    reward -= 2.0
                else:
                    reward += 0.0
    return reward

def neg_rew(predict_str: str) -> float:
    # Match fenced code blocks of json, xml, or html
    if re.search(r"```json", predict_str, re.IGNORECASE) \
       or re.search(r"```xml", predict_str, re.IGNORECASE) \
       or re.search(r"```html", predict_str, re.IGNORECASE):
        return -5.0
    return 0.0
    
def perception_tool_usage_reward(predict_str: str) -> float:
    # Check for perception tool usage
    perception_calls = re.findall(r'<tool_call>.*?"name"\s*:\s*"perception_tool".*?</tool_call>', predict_str, re.DOTALL)

    if perception_calls:
        return 10.0  # Reward for using perception tool
    else:
        return -1.0  # Penalty for not using it when expected
def no_tool_usage_penalty(predict_str: str) -> float:
    # Check for any tool usage
    tool_calls = re.findall(r'<tool_call>.*?"name"\s*:\s*"(python_tool|ocr_tool|detection_tool|captioning_tool|perception_tool|search_tool)".*?</tool_call>', predict_str, re.DOTALL)

    if not tool_calls:
        return -8.0  # Penalty for not using any tool
    else:
        return 0.0  # No penalty if tools are used
# -------------------------------------------------------
# Final Score Computation
# -------------------------------------------------------
def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    format_score: float = 0.2
):
    correct, pred, num_answers = correct_answer_reward(solution_str, ground_truth)

    if correct:
        if num_answers == 0:
            acc_reward = 5.0
        else:
            acc_reward = 10.0 / 1.0
    else:
        acc_reward = -2.0

    abs_max = 8.0 + 8.0 + 1.5 + 6.0 + 10.0 #7.5   # Normalizing factor

    #fmt_rew = format_reward(solution_str)
    #tool_fmt_rew = correct_tool_format_reward(solution_str)
    strict_ans_rew = strict_format_reward(solution_str)
    spc_fmt = space_format_reward(solution_str)
    tool_pres_rew = tool_call_presence_reward(solution_str)
    perception_tool_rew = perception_tool_usage_reward(solution_str)
    no_tool_penalty = no_tool_usage_penalty(solution_str)
    #empty_penalty = empty_tool_format_reward(solution_str)
    #neg_penalty = neg_rew(solution_str)
    
    pattern_success = re.compile(r"<tool_response>\s*\{\s*\"success\"\s*:\s*true.*?\}\s*</tool_response>",re.DOTALL)
    if pattern_success.search(solution_str):
        tool_success_rew= 6.0
    else:
        tool_success_rew= -2.0

    # Determine if tool was used
    #tool_used = bool(pattern_alt.search(solution_str)) and bool(pattern_success.search(solution_str))
    # if tool_used and correct and num_answers ==1:
    #     final_score = 5.0
    # else:
    #tool_called_true = bool(re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL).search(solution_str))
    #perception_calls = bool(re.compile(r'<tool_call>.*?"name"\s*:\s*"perception_tool".*?</tool_call>', re.DOTALL).search(solution_str))
    
    # if correct and num_answers >= 1 and perception_calls:
    #     final_score = 2.4 
    # if correct and num_answers >= 1 and tool_called_true:
    #     final_score = 2.4
    # elif correct and num_answers >= 1:
    #     final_score = 2.0
    # elif correct and num_answers == 0 and tool_called_true:
    #     final_score = 1.7    
    # elif correct and num_answers == 0:
    #     final_score = 1.2
    # else:
    if num_answers == 0:
        final_score = 0.0
    else:
        final_score = 1.0 * ( 1* acc_reward +  2 * strict_ans_rew + 0 * spc_fmt + 1*tool_pres_rew + 0*tool_success_rew + 0*perception_tool_rew + 1*no_tool_penalty) / abs_max

    return {
        "score": final_score,
        "acc": correct,
        "pred": pred,
    }
