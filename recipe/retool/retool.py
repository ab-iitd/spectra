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
import logging
import re
from typing import Any, Optional
from uuid import uuid4

import datasets

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.sandbox_image_tools import SandboxImageTools
from verl.tools.sandbox_web_search_tools import SandboxWebSearchTools
from verl.tools.schemas import ToolResponse
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo, spectra_perception
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomToolSwitcher(BaseTool):
    """A unified tool that switches between code execution and image captioning based on tool name."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.name = tool_schema.function.name if hasattr(tool_schema, 'function') and hasattr(tool_schema.function, 'name') else ""
        
        # Initialize the appropriate tool based on the name
        if self.name == "python_tool":
            self.underlying_tool = CustomSandboxFusionTool(config, tool_schema)
        elif self.name == "captioning_tool":
            self.underlying_tool = SandboxImageTools(config, tool_schema)
        elif self.name == "detection_tool":
            self.underlying_tool = SandboxImageTools(config, tool_schema)
        elif self.name == "ocr_tool":
            self.underlying_tool = SandboxImageTools(config, tool_schema)
        elif self.name == "perception_tool":
            self.underlying_tool = SandboxImageTools(config, tool_schema)
        elif self.name == "web_search_tool":
            self.underlying_tool = SandboxWebSearchTools(config, tool_schema)
        # Will be adding custom functions here 
        
        else:
            raise ValueError(f"Unknown tool name: {self.name}")
        
        logger.info(f"Init CustomToolSwitcher with name: {self.name}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.underlying_tool.get_openai_tool_schema()

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        return await self.underlying_tool.create(instance_id, ground_truth, **kwargs)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await self.underlying_tool.execute(instance_id, parameters, **kwargs)

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return await self.underlying_tool.calc_reward(instance_id, **kwargs)

    async def release(self, instance_id: str, **kwargs) -> None:
        await self.underlying_tool.release(instance_id, **kwargs)


class CustomSandboxFusionTool(SandboxFusionTool):
    """Custom code execution tool with enhanced code processing."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


answer_format = """ """


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # Explicitly specify the parquet loader
            dataframe = datasets.load_dataset("parquet", data_files={"train": parquet_file})["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "yentinglin/aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content #+ answer_format
        row["agent_name"] = "tool_agent"
        return row


def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # # encourage model to call tools
    # num_turns = extra_info["num_turns"]
    # if result["score"] < 0:
    #     tool_call_reward = 0*(num_turns - 2) / 2 * 0.1
    #     result["score"] = min(0, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result