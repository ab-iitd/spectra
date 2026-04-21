set -x
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
#export VERL_LOGGING_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0,1  # Specify your devices here
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =================== wandb ===================
export WANDB_API_KEY='Your Custom API Key Here'
project_name=Qwen2.5VL_7B_RL
experiment_name=Qwen2.5VL_7B_RL
default_local_dir=/verl/spectra/$experiment_name

# =================== logging ===================
EXPERIMENT_NAME="Qwen2.5VL_7B_RL_full_eval_$(now)"
RESULTS_DIR="/verl/spectra/$EXPERIMENT_NAME"

# Create results directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR/rollout_data"
mkdir -p "$RESULTS_DIR/mem_snapshots"

exec > >(tee "$RESULTS_DIR/evaluation.log") 2>&1
# ================= Parameters =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.00
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2 
clip_ratio_high=0.4

max_turns=16
max_prompt_length=16384
max_response_length=2048
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=32
ppo_micro_batch_size=16  

n_resp_per_prompt=8
n_resp_per_prompt_val=1

top_p=0.95
val_top_p=0.70
top_k=-1
temperature=1.0
val_temperature=1.0

backend=${BACKEND:-fsdp}

FSDP_ENGINE_CONFIG="\
    optim.betas="[0.9,0.99]" \
    optim.warmup_style=cosine \
"
# ================= perfomance =================
infer_tp=1 # vllm
train_sp=1 # train
offload=False  

CONFIG_PATH="/verl/spectra/config"

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='chat_template' \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files=/verl/spectra/data/Your_custom_Training_data.parquet \
    data.val_files=/verl/spectra/data/Your_custom_Validation_data.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.return_multi_modal_inputs=False \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    data.truncation=right \
    data.image_key=images \
    data.custom_cls.path=/verl/spectra/spectra_tool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=/verl/spectra/spectra_tool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=/""Your Model Path""/Qwen2.5VL-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj] \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.weight_decay=0.1\
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_micro_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/verl/spectra/config/sandbox_fusion_tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=/verl/spectra/config/chat_template.yaml \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.72 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name='geo3k_async_rl' \
    trainer.experiment_name='qwen2.5vl_7b_function_rm2' \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    trainer.critic_warmup=20 \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=/verl/spectra/Qwen2.5_VL/Training_op \
    trainer.validation_data_dir=/verl/spectra/Qwen2.5_VL/Validation_op \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=5 \
    trainer.save_freq=100 \
    trainer.total_epochs=2 $@

echo "Evaluation completed. Results saved to: $RESULTS_DIR"
echo "Key files:"
echo "- Full log: $RESULTS_DIR/evaluation.log"
echo "- Rollout data: $RESULTS_DIR/rollout_data/"
echo "- Memory snapshots: $RESULTS_DIR/mem_snapshots/"
echo "- Checkpoints: $RESULTS_DIR/checkpoints/"
