# Copyright 2025 Andrew Duncan and Paula Cordero Encinar

from typing import List
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt_entropy(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, entropy_list = _batch_majority_vote_entropy(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i] # Return the values that we need to compute the loss function (it is a list)
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i][0] # Return the majority vote answer
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["negative_entropy_list"] = np.array(entropy_list, dtype=float)
    return batch


def _batch_majority_vote_entropy(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        entropy_list: list of float with values for the negative entropy
    """
    majority_gt_list = []
    entropy_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]

        prompt_majority_gt, prompt_entropy = _majority_vote_entropy(prompt_outputs)

        majority_gt_list.append(prompt_majority_gt)
        entropy_list.append(prompt_entropy)

    return majority_gt_list, entropy_list


def _majority_vote_entropy(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return ["None", 0.0, {}], -100
    
    counts = Counter(model_answers)
    
    majority_answer, majority_count = counts.most_common(1)[0]

    entropy_estimator = compute_entropy(counts, alpha=0.5)

    return [majority_answer, majority_count, counts], entropy_estimator


# === Metrics Computation ===


def compute_ttrl_metrics_entropy(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics_entropy(majority_reward, gt_reward, majority_label, gt_label, n=n)
    negative_entropy_list = batch.non_tensor_batch["negative_entropy_list"]
    negative_entropy = sum(negative_entropy_list) / len(negative_entropy_list)
    ttrl_metrics["negative_entropy"] = negative_entropy

    return ttrl_metrics


def _batch_compute_ttrl_metrics_entropy(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics_entropy(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics_entropy(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0       
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric

# === Auxiliary Function ===

def compute_entropy(counts, alpha = 0.5):

    n_total = sum(N for N in counts.values())
    entropy = sum((N+alpha)/(n_total+alpha) * np.log((N+alpha)/(n_total+alpha)) for N in counts.values() if N > 0)

    return entropy
