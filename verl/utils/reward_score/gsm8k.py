# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def format_reward(solution_str: str) -> float:
    """
    Matches tags and gives reward score based on that
    Matches : 
        * <think>  * <image_understanding>  * <reflection>  * <answer>
    """
    reward=0
    
    # Reward for tags 
    # ARTIST apper like reward structure (Relaxed rewards)
    think = re.compile(r"<think>.*?</think>", re.DOTALL)
    reflection = re.compile(r"<reflection>.*?</reflection>", re.DOTALL)
    answer = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    tool_call = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
    boxed = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
    #order_pattern = re.compile(r"<think>.*?</think>\s*<reflection>.*?</reflection>\s*<answer>.*?</answer>",re.DOTALL)

    if re.search(think, solution_str): 
        reward += 0.125
    if re.search(reflection, solution_str): 
        reward += 0.125
    if re.search(answer, solution_str): 
        reward += 0.125
    if re.search(tool_call, solution_str): 
        reward += 0.125
    if re.search(boxed, solution_str): 
        reward += 0.125
    return max(reward, 0.0)

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0 + 0.1* format_reward(solution_str)
    else:
        if answer == ground_truth:
            return score +0.1* format_reward(solution_str)
        else:
            return format_score +0.1* format_reward(solution_str)
