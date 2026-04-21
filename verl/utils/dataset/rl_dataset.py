import copy
import logging
import os
from collections import defaultdict
from typing import Optional, Any

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )

                    images, videos = self._extract_images_videos(doc, messages)

                    if images is not None:
                        images = [process_image(im) for im in images]
                    if videos is not None:
                        videos = [process_video(v) for v in videos]

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        # IMPORTANT: prompt is typically a list[{"role":..., "content":...}, ...]
        messages: list = example.pop(self.prompt_key)
        return messages

    def _extract_images_videos(self, row_dict: dict, messages: list):
        """
        Robust extraction:
        1) row_dict["images"] (default) or row_dict["image"] fallback
        2) messages content list items of type image/video (OpenAI-style)
        """
        images = None
        videos = None

        # ---- 1) Try row_dict columns ----
        # image_key (default "images")
        if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
            images = row_dict.get(self.image_key)
        # fallback to singular "image"
        elif "image" in row_dict and row_dict.get("image", None) is not None:
            images = row_dict.get("image")

        if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
            videos = row_dict.get(self.video_key)
        elif "video" in row_dict and row_dict.get("video", None) is not None:
            videos = row_dict.get("video")

        # Normalize: some datasets store a single item not list
        if images is not None and not isinstance(images, list):
            images = [images]
        if videos is not None and not isinstance(videos, list):
            videos = [videos]

        # ---- 2) If still missing, extract from messages ----
        if (images is None or len(images) == 0) or (videos is None or len(videos) == 0):
            msg_images = []
            msg_videos = []
            for m in messages:
                c = m.get("content", None)
                if isinstance(c, list):
                    for part in c:
                        if not isinstance(part, dict):
                            continue
                        ptype = part.get("type", None)

                        # image formats: {"type":"image","image":...} OR {"type":"image","image_url":...}
                        if ptype == "image" or ("image" in part) or ("image_url" in part):
                            msg_images.append(part)

                        # video formats: {"type":"video","video":...} OR {"type":"video","video_url":...}
                        if ptype == "video" or ("video" in part) or ("video_url" in part):
                            msg_videos.append(part)

            if images is None or len(images) == 0:
                images = msg_images if len(msg_images) > 0 else images
            if videos is None or len(videos) == 0:
                videos = msg_videos if len(msg_videos) > 0 else videos

        # Final normalize
        if images is not None and not isinstance(images, list):
            images = [images]
        if videos is not None and not isinstance(videos, list):
            videos = [videos]

        return images, videos

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)

        from verl.utils.dataset.vision_utils import process_image, process_video

        # ---- Extract images/videos robustly BEFORE building prompt ----
        images_raw, videos_raw = self._extract_images_videos(row_dict, messages)

        # Pop column images/videos if present, to avoid carrying huge blobs twice
        # (keep the extracted copies we already have)
        if self.image_key in row_dict:
            row_dict.pop(self.image_key, None)
        row_dict.pop("image", None)

        if self.video_key in row_dict:
            row_dict.pop(self.video_key, None)
        row_dict.pop("video", None)

        images = None
        videos = None
        multi_modal_data = {}

        if images_raw is not None and len(images_raw) > 0:
            images = [process_image(im) for im in images_raw]
            multi_modal_data["image"] = images

        if videos_raw is not None and len(videos_raw) > 0:
            videos = [process_video(v) for v in videos_raw]
            multi_modal_data["video"] = [v.numpy() for v in videos]

        row_dict["multi_modal_data"] = multi_modal_data

        # ---- Build raw prompt (template decides <image> vs vision tokens) ----
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )

            # DEBUG (keep your print style)
            has_image_token = ("<image>" in raw_prompt) or ("<|vision_start|>" in raw_prompt)
            num_images = 0 if images is None else len(images)
            print(f"INTERNVL raw_prompt has <image>: {('<image>' in raw_prompt)} num_images: {num_images}")

            if has_image_token and num_images == 0:
                # This is the exact failure mode you're seeing
                print("WARNING: Prompt contains <image> token but no images were found in row_dict/messages.")
                print("Available keys in sample (after prompt pop):", list(row_dict.keys()))
                print("Hint: ensure parquet has an images column or messages contain image parts.")

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            # Remove if present
            model_inputs.pop("second_per_grid_ts", None)

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # ---- Postprocess/pad ----
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # ---- Position ids ----
        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict.get("data_source"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()