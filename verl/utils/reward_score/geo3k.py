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
from mathruler.grader import extract_boxed_content, grade_answer

# def format_reward(predict_str: str) -> float:
#     pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
#     match_result = re.fullmatch(pattern, predict_str)
#     return 1.0 if match_result else 0.0

#______________________________________________________________________________________#
# Format reward like ARTIST  
def format_reward(predict_str: str) -> float:
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

    if re.search(think, predict_str): 
        reward += 0.125
    if re.search(reflection, predict_str): 
        reward += 0.125
    if re.search(answer, predict_str): 
        reward += 0.125
    if re.search(tool_call, predict_str): 
        reward += 0.125
    if re.search(boxed, predict_str): 
        reward += 0.125

    # Strict format reward for having all the tags
    # if re.search(order_pattern, predict_str):
    #     if all(re.search(p, predict_str) for p in [think, reflection, answer, boxed]):
    #         reward += 0.5

    # # Penalties
    # if predict_str.count("<think>") > 2: 
    #     reward -= 0.25
    # if re.search(think, predict_str) and len(re.search(think, predict_str).group(0)) < 15:
    #     reward -= 0.25
    # if re.search(boxed, predict_str) and not re.search(answer, predict_str):
    #     reward -= 0.5

    return max(reward, 0.0)
#______________________________________________________________________________________#
# Accuracy reward. Reward for matching the answer with the ground truth
def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 2.0 if grade_answer(answer, ground_truth) else 0.0
#______________________________________________________________________________________#
def tool_call_reward(predict_str:str):
    pass


#______________________________________________________________________________________#
def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(predict_str)
