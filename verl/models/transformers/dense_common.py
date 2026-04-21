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

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def _resolve_decoder_and_lm_head(self):
    """
    Resolve (decoder_model, lm_head) for different model wrappers.

    Common HF CausalLMs:
      - self.model is the decoder (e.g., LlamaModel)
      - self.lm_head is the vocab projection

    InternVL-like wrappers (e.g., InternVLChatModel):
      - self.language_model or self.llm is often a *CausalLM* (e.g., LlamaForCausalLM)
      - decoder is language_model.model / llm.model
      - head is language_model.lm_head / llm.lm_head
    """
    # Fast path: standard HF-style (LlamaForCausalLM, etc.)
    if hasattr(self, "model") and hasattr(self, "lm_head"):
        return self.model, self.lm_head

    # Try common wrapper attributes used by VLMs / InternVL
    wrapper_candidates = []
    for attr in ["language_model", "llm", "model_llm", "text_model", "base_model"]:
        if hasattr(self, attr):
            wrapper_candidates.append(getattr(self, attr))

    for wrapper in wrapper_candidates:
        if wrapper is None:
            continue

        # Many wrappers store a CausalLM-like module that itself has .model and .lm_head
        if hasattr(wrapper, "model") and hasattr(wrapper, "lm_head"):
            return wrapper.model, wrapper.lm_head

        # Some wrappers store the decoder directly (rare)
        if hasattr(wrapper, "layers") or hasattr(wrapper, "embed_tokens"):
            # If decoder-like, try to find lm_head from self or wrapper
            if hasattr(self, "lm_head"):
                return wrapper, self.lm_head
            if hasattr(wrapper, "lm_head"):
                return wrapper, wrapper.lm_head

    # Last attempt: sometimes InternVL puts the causal LM under `self.model` but without lm_head
    if hasattr(self, "model") and not hasattr(self, "lm_head"):
        m = getattr(self, "model")
        if hasattr(m, "model") and hasattr(m, "lm_head"):
            return m.model, m.lm_head

    # If nothing worked, fail loudly with useful debugging info
    attrs = sorted(list(self.__dict__.keys()))
    raise AttributeError(
        f"[dense_common] Cannot resolve decoder/lm_head for model class={self.__class__.__name__}. "
        f"Expected either (self.model + self.lm_head) or wrapper like (self.language_model/.llm with .model + .lm_head). "
        f"Available instance attrs (partial): {attrs[:80]}"
    )


def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> CausalLMOutputWithPast:
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # IMPORTANT: resolve decoder (not the top wrapper) for InternVL-like models
    decoder_model, _ = _resolve_decoder_and_lm_head(self)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = decoder_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    return outputs


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    # We REQUIRE return_dict=True because we access outputs.past_key_values/hidden_states/attentions below
    if return_dict is None:
        return_dict = True

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # For decoder models (e.g., LlamaModel), outputs[0] is hidden_states
    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    # IMPORTANT: lm_head might live under wrapper (InternVL) -> resolve it
    _, lm_head = _resolve_decoder_and_lm_head(self)

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    if return_dict is None:
        return_dict = True

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    _, lm_head = _resolve_decoder_and_lm_head(self)

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )