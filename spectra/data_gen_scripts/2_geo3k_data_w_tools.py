## GEO3K data Generation
import os
import datasets

local_dir = "/home2/data/geo3k_unprocessed"
hdfs_dir = None  # Can set if needed
data_source = "hiyouga/geometry3k"

instruction_following = ""  # Custom instruction to follow

# Loading dataset
dataset = datasets.load_dataset(data_source)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

if "validation" in dataset:
    val_dataset = dataset["validation"]
else:
    split_result = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_result["train"]
    val_dataset = split_result["test"]

# For Generating specific subsets of the dataset like 500 or 1000 datapoints
#train_dataset = train_dataset.select(range(min(500, len(train_dataset))))
#test_dataset = test_dataset.select(range(min(15, len(test_dataset))))
#val_dataset = val_dataset.select(range(min(15, len(val_dataset))))

# Processing function for the data
def make_map_fn(split):
    def process_fn(example, idx):
        problem = example.pop("problem")
        answer = example.pop("answer")
        images = example.pop("images")

        # Adding image link/ id column as unique identifier for tool usage
        image_link = f"geo3k/{split}_image_{idx+1:02d}"

        prompt_text = (
            problem
            + " "
            + f" The image URL for this question: {image_link}."
        )

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": (
    r"""You are a helpful assistant that can solve complex math problems step by step, optionally using image understanding and a Python execution tool.
Instructions:
    For each question, think through the solution by using one or both reasoning categories:
    <think_perception>Use this tag to show reasoning related to employing perceptive tools such as captioning_tool, detection_tool, or ocr_tool to extract information from a provided image. </think_perception>
    <think_reason> Use this tag to show reasoning behind solving the task based on available information or responses from perceptive tools and/or the Python execution tool. </think_reason> 
    Enclose the output of reasoning using the appropriate tags as specified.
    Enclose only the final answer within <answer> {Final Answer Here} </answer>.
    You can call tools to help you:
    Perception tools (for understanding a provided image URL/ID): captioning_tool, detection_tool, ocr_tool
    Code execution tool (for running Python): code_interpreter (Always end code with a print statement print())
Tool Calls:
    Use them when you need to understand or extract information from the provided image or execute Python code for calculations, simulations, or symbolic math. Allowed libraries: numpy, scipy, sympy, time, random. Always print the final result at the end of your Python code so it appears in the tool result message. Each execution is independent; variables do not persist across separate calls.
    Call format:
    <tool_call> {name: {captioning_tool|detection_tool|ocr_tool|code_interpreter}, arguments: {image URL or ID}} </tool_call>
    Result format (returned to you by the tool in appropriate format)
General guidance:
    You may make multiple iterations of relevant tool calls to arrive at the answer.
    If a tool call fails (success: False), adjust your approach and try again if appropriate.
    Pick perceptive tools based on <think_perception> reasoning and solve the task based on <think_reason> reasoning.
    When an image URL/ID is provided, use perception tools as needed to extract information before proceeding with calculations.
    Place tool calls and their results outside the reasoning tags or final <answer> {Final Answer Here} </answer> tags.
    All the thinking ,reasoning and answer generation should be within <the specified tags before> <think_perception> </think_perception> or <think_reason> </think_reason> <tool_call> </tool_call> or <answer> </answer> Tags.
    make sure to use these tags as they help you to think.  
"""
                    ),

                },
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ],
            "images": images,
            "image_link": [image_link],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": problem,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "calc_geo3k_reward": {
                        "create_kwargs": {"ground_truth": answer},
                    },
                },
            },
        }
        return data

    return process_fn

# Map preprocessing over all splits
train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True, num_proc=8)
test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

# Save to parquet
os.makedirs(local_dir, exist_ok=True)
train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

if hdfs_dir is not None:
    makedirs(hdfs_dir)
    copy(src=local_dir, dst=hdfs_dir)

print(f"Datasets saved in {local_dir}")
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")
