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
                    "content": (r""
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
