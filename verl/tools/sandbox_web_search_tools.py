import logging
import os
import threading
import json
import requests
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            if self.rate_limit_worker:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SandboxWebSearchTools(BaseTool):
    """A tool for sending a web search query to a sandbox environment for processing."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        logger.info(f"Init SandboxImageTools with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        timeout = parameters.get("timeout", self.default_timeout)
        if not isinstance(query, str):
            query = str(query)

        result = await self.execution_pool.execute.remote(self.execute_image_tools, instance_id, query, timeout)
        return ToolResponse(text=result.text), None, None

    def execute_image_tools(self, instance_id, query, timeout=30):
        try:
            payload = {"query": query, "timeout": timeout}
            resp = requests.post(self.sandbox_fusion_url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                return ToolResponse(text=f"Error: HTTP {resp.status_code}")
            data = resp.json()
            if data.get("success"):
                caption = json.dumps(data)
                logger.debug(f"Search result for {instance_id}: {caption}")
                return ToolResponse(text=caption)
            else:
                return ToolResponse(text="Search tool failed: " + json.dumps(data))
        except Exception as e:
            logger.warning(f"Error contacting container: {e}")
            return ToolResponse(text=f"Exception during search tool execution: {e}")

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
