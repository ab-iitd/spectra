import re
from typing import Optional
from collections import Counter
# REGEX patterns
# This is for proper Image tools execution    # 1.0

# pattern1 = re.compile(r"""    # tool call -> tool response -> think perception -> optional last think reasoning -> answer
# (
#     (?:<think_reasoning>.*?</think_reasoning>\s*
#         (?: 
#             (?:<tool_call>\s*
#                 \{\s*"name"\s*:\s*"(?:captioning_tool|perception_tool|ocr_tool|detection_tool)"\s*,\s*
#                 "arguments"\s*:\s*\{.*?\}\s*
#                 \}
#              \s*</tool_call>\s*
#              <tool_response>.*?</tool_response>\s*)+
#              <think_perception>.*?</think_perception>\s*
#         )+
#         <think_reasoning>.*?</think_reasoning>\s*
#     )+
#     <answer>.*?</answer>\s*
# )
# """, re.VERBOSE | re.DOTALL)

# # This is for proper PYTHON TOOL execution    # 1.0
# pattern2 = re.compile(r"""  
# (
#     (?:
#         <think_reasoning>.*?</think_reasoning>\s*
#         (?:
#             <tool_call>\s*
#             \{\s*"name"\s*:\s*"code_interpreter"\s*,\s*
#             "arguments"\s*:\s*\{\s*"code"\s*:\s*"([^"]*)"\s*\}\s*
#             \}
#             \s*</tool_call>\s*
#             <tool_response>.*?</tool_response>\s*
#             <think_reasoning>.*?</think_reasoning>\s*
#         )+
#     )+
#     <answer>.*?</answer>\s*
# )
# """, re.VERBOSE | re.DOTALL)


# This is for ANY SORT OF ORDER     # 1.0 
# This is for ANY SORT OF ORDER (strict superset)  # 1.0
pattern3 = re.compile(r"""
(
    (?:<think_reasoning>.*?</think_reasoning>\s*
        (?:
            # Perception or OCR tools
            (?:
                <tool_call>\s*
                \{\s*"name"\s*:\s*"(?:captioning_tool|perception_tool|ocr_tool|detection_tool)"[^}]*\}
                \s*</tool_call>\s*
                <tool_response>.*?</tool_response>\s*
                <think_perception>.*?</think_perception>\s*
                <think_reasoning>.*?</think_reasoning>\s*
            )
            |
            # Python tool / code interpreter
            (?:
                (?:<tool_call>\s*
                    \{\s*"name"\s*:\s*"(?:python_tool|code_interpreter)"[^}]*\}
                 \s*</tool_call>\s*
                 <tool_response>.*?</tool_response>\s*
                 <think_reasoning>.*?</think_reasoning>\s*)+
            )
        )+
    )+
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL)


pattern4 = re.compile(r"""  # 0.5
(
    <think_reasoning>.*?</think_reasoning>\s*
    <think_perception>.*?</think_perception>\s*
    <think_reasoning>.*?</think_reasoning>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL)

pattern6 = re.compile(r"""  # 0.5
(
    <think_reasoning>.*?</think_reasoning>\s*
    <tool_call>.*?</tool_call>\s*
    <tool_response>.*?</tool_response>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL)

pattern5 = re.compile(r"""    # 0.25
(
    <think_reasoning>.*?</think_reasoning>\s*
    <answer>.*?</answer>\s*
)
""", re.VERBOSE | re.DOTALL)


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
    "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to keep only digits, decimals, frac{}, sqrt{}."""
    final_answer = final_answer.split("=")[-1]  # RHS side if equation in ans tags

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)

    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # LaTeX commands and formatting
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"\3", final_answer)
    final_answer = re.sub(r"\\text\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\textbf\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\overline\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"\\boxed\{(.*?)\}", r"\1", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = re.sub(r"[\[\]\{\}\(\)]", "", final_answer)

    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").replace(".", "").isdigit():
        final_answer = final_answer.replace(",", "")

    allowed_parts = re.findall(r"(?:\d+\.\d+|\d+|frac\{\d+\}\{\d+\}|sqrt\{\d+\})", final_answer)
    final_answer = "".join(allowed_parts)

    final_answer = re.sub(r"(\d+)\.0$", r"\1", final_answer)

    if "sqrt" in final_answer:
        return final_answer.strip()

    return final_answer.strip()


# Reward design
# -------------------------------------------------------
# 1. CORRECT ANSWER REWARD
def correct_answer_reward(solution_str: str, ground_truth: str) -> tuple[bool, str]:
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    pred = normalize_final_answer(match.group(1)) if match else ""
    gt = ground_truth.strip()
    return (pred == gt), pred

# 2. FORMAT REWARD WITH LENGTH PENALTY
def format_reward(predict_str: str) -> float:
    rw_f = 1
    lp = 0.5

    if pattern3.search(predict_str):
        return rw_f
    elif pattern4.search(predict_str):
        return rw_f * lp
    elif pattern5.search(predict_str) or pattern6.search(predict_str):
        return rw_f * lp / 2
    return 0.0

# 3. TOOL CALL FORMAT REWARD
def correct_tool_format_reward(predict_str: str) -> float:
    tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)
    reward = 0.0
    #______________________________________________________
    # CORRECT TOOL CALL FORMATTING
    for call in tool_calls:
        name = re.search(r'"name"\s*:\s*"(code_interpreter|ocr_tool|detection_tool|captioning_tool|perception_tool)"',call)
        arguments= re.search(r'"arguments"\s*:\s*(\{.*?\}|"(?:\\.|[^"])*")',call,re.DOTALL)
        if name and arguments:
            reward += 0.1   # each valid pair contributes 0.05
    #______________________________________________________
    
    allowed_tools = {"code_interpreter", "ocr_tool", "detection_tool", "captioning_tool", "perception_tool"}
    tool_usage = Counter()
    
    for call in tool_calls:
        match = re.search(r'"name"\s*:\s*"([^"]+)"', call)
        if match:
            tool_name = match.group(1)
            if tool_name in allowed_tools:
                tool_usage[tool_name] += 1
                
    # Check if any tool is used more than 2 times
    if all(count <= 3 for count in tool_usage.values()):
        reward += 0.5
    else:
        reward -= 0.25
    #______________________________________________________
    # Checking for duplicate tool calls    
    unique_tool_calls = set(tool_calls)  
    if len(tool_calls) == len(unique_tool_calls):
        reward += 0.5
    else:
        reward -= 0.25
    return min(reward,1.30)

# 4. STRICT think_reasoning at the beginning and answer tag at the end
def strict_format_reward(predict_str: str) -> float:
    answer_at_end = re.search(r"<answer>.*?</answer>\s*$", predict_str, re.DOTALL)
    ans_tags = re.findall(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    
    if answer_at_end and len(ans_tags) == 1 :
        return 1.5 
    else:
        return -0.3



# def tool_call_cap(predict_str: str) -> float:
#     reward = 0
#     tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)
#     allowed_tools = {"code_interpreter", "ocr_tool", "detection_tool", "captioning_tool", "perception_tool"}

#     tool_usage = Counter()
    
#     for call in tool_calls:
#         match = re.search(r'"name"\s*:\s*"([^"]+)"', call)
#         if match:
#             tool_name = match.group(1)
#             if tool_name in allowed_tools:
#                 tool_usage[tool_name] += 1
                
#     # Check if any tool is used more than 2 times
#     if all(count <= 2 for count in tool_usage.values()):
#         reward += 0.5
        
#     # Checking for duplicate tool calls    
#     unique_tool_calls = set(tool_calls)  
#     if len(tool_calls) == len(unique_tool_calls):
#         reward += 0.5
    
#     return min(reward,1.0)


# # Tool call success reward
# def correct_tool_call_reward(predict_str: str) -> float:
#     tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>",predict_str, re.DOTALL)
#     tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>",predict_str, re.DOTALL)
    
#     reward = 0.0
#     total_call_count= len(tool_calls)
#     tool_response_count = len(tool_responses)
#     if tool_calls and tool_responses and total_call_count == tool_response_count:  # if tool call and tool response number is equal
#         reward = 0.5

#     if total_call_count > 0:  # Reward on tool success ratio
#         success_responses = sum(1 for r in tool_responses if '"success": "True"' in r)
#         tool_success_ratio = 1.0 * success_responses / total_call_count
#         reward += tool_success_ratio
#     return reward



# def tool_call_vs_tool_response_reward(predict_str: str) -> float:
#     tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", predict_str, re.DOTALL)
#     tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", predict_str, re.DOTALL)
    
#     if len(tool_calls) == len(tool_responses) and len(tool_calls) > 0:
#         return 1.0
#     else:
#         return 0.0

# Final score computation
# -------------------------------------------------------
def compute_score(solution_str: str,
                  ground_truth: str,
                  strict_box_verify: bool = False,
                  pause_tokens_index: Optional[list[int]] = None,
                  format_score: float = 0.2):
    correct, pred = correct_answer_reward(solution_str, ground_truth)
    acc_reward = 5.0 if correct else 0.0
    
    abs_max = (8.8 + 1.0)/3  #----> Normalizing factor

    fmt_rew = format_reward(solution_str)
    tool_fmt_rew = correct_tool_format_reward(solution_str)
    strict_ans_rew = strict_format_reward(solution_str)
    #correct_tool_rew = correct_tool_call_reward(solution_str)
    #duplicate_penalty = no_duplicate_tool_call_reward(solution_str)
    #tool_call_tool_resp_rew = tool_call_vs_tool_response_reward(solution_str)
    
    
    final_score = (acc_reward + fmt_rew + tool_fmt_rew + strict_ans_rew) / abs_max 

    return {
        "score": final_score,
        "acc": correct,
        "pred": pred,
    }
