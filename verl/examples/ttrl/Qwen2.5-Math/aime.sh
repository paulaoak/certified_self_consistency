#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export WANDB_API_KEY=7dbc60219e2e293affe5503a11b96fbc218bf5cb

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AIME-TTT"
BACKBONE="Qwen2.5-Math-1.5B"
ADVANTAGE="grpo"
REWARD_snr=False
REWARD_entropy=True

if [  "$REWARD_snr" = True ] && [ "$REWARD_entropy" = True ]; then
  echo "Please enable only one reward function."
  exit 1
fi

if [ "$REWARD_snr" = True ]; then
  REWARD_FUNC=reward_func
  REWARD_NAME="SNR"
elif [ "$REWARD_entropy" = True ]; then
  REWARD_FUNC=reward_func_entropy
  REWARD_NAME="ENTROPY"
else
  echo "Please enable at least one reward function."
  exit 1
fi

K=3
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * $K))
if [ "$K" -gt 8 ]; then
  N=4
else
  N=16
fi

EPISODE=80
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=64
N_SAMPLES_PER_PROMPT=32
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

DATA_LOCAL_DIR="data"
BACKBONE_PATH="Qwen/Qwen2.5-Math-1.5B"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-Len@${K}k"

WANDB_PROJECT="CONDORCET-AIME"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}-${REWARD_NAME}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.dtype='bfloat16' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  +actor_rollout_ref.rollout.val_kwargs.n_last=100 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  +critic.model.dtype='bfloat16' \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=2 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  +reward.snr_based_reward=$REWARD_snr \
  +reward.entropy_based_reward=$REWARD_entropy \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=$REWARD_FUNC \
  +custom_reward_function.name_val=reward_func_val \
  +ttt.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=2000000 \
  trainer.test_freq=2 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"
 
echo "Output directory: $OUTPUT_DIR"
