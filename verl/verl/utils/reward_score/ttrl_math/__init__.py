# Copyright 2025 Paula Cordero Encinar and Andrew Duncan
#

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

from latex2sympy2_extended import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
import traceback
from collections import Counter
import numpy as np
import warnings

from .math_utils import extract_boxed_answer, is_latex_equal, grade_answer_mathd, grade_answer_sympy, timeout_ours

"""
This code is adapted from Entropy Mechanism Recipe (https://github.com/volcengine/verl/tree/main/recipe/entropy/).
"""

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct

@timeout_ours(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    try:
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except TimeoutError:
        return expression_string
    except Exception as e:
        try:
            sympy_expr = latex2sympy(expression_string)
            simplified_expr = simplify(sympy_expr)
            return str(simplified_expr)
        except TimeoutError:
            return expression_string
        except Exception as e:
            return expression_string
        
# === Training score for SNR reward ===
        
def compute_score(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)
    majority_vote = gt_answer[0]

    if model_answer is None:
        return {
            "score": -10.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": majority_vote,
            "pred": "",
        }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct_majority = False
    is_correct_runner_up = False
    
    runner_up = gt_answer[1]
    majority_count = float(gt_answer[2])
    runner_up_count = float(gt_answer[3])
    total_count = float(gt_answer[4])

    if isinstance(majority_vote, float) or isinstance(majority_vote, int):
        majority_vote = str(majority_vote)

    if isinstance(runner_up, float) or isinstance(runner_up, int):
        runner_up = str(runner_up)

    if isinstance(majority_vote, str):
        is_correct_majority = grade(model_answer, majority_vote, fast)
    if isinstance(runner_up, str):
        is_correct_runner_up = grade(model_answer, runner_up, fast)
       
    if is_correct_majority:
        score = compute_SNR(majority_count, runner_up_count, total_count) - compute_SNR(majority_count - 1, runner_up_count, total_count - 1)
        return {
            "score": score,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": majority_vote,
            "pred": model_answer,
        }
    
    elif is_correct_runner_up:
        score = compute_SNR(majority_count, runner_up_count, total_count) - compute_SNR(majority_count, runner_up_count - 1, total_count - 1)
        return {
            "score": score,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": majority_vote,
            "pred": model_answer,
        }
    else:
        score = compute_SNR(majority_count, runner_up_count, total_count) - compute_SNR(majority_count, runner_up_count, total_count - 1)
        return {
            "score": score,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": majority_vote,
            "pred": model_answer,
        }

def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = compute_score(solution_str, ground_truth)

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise

# === Training score for Entropy reward ===
def compute_score_entropy(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)

    majority_vote = gt_answer[0]
    counts = gt_answer[2]

    if isinstance(counts, str):
        warnings.warn(
            f"Invalid type for counts: got str ('{counts}'). "
            "It must be a dict, not a str. Proceeding with empty dict instead.",
            UserWarning
        )
        counts = {}

    if model_answer is None:
        return {
            "score": -10.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": majority_vote,
            "pred": "",
        }
        # return 0.0, 0.0  # Cannot even parse anything.

    counts_copy = counts.copy()

    if isinstance(majority_vote, float) or isinstance(majority_vote, int):
        majority_vote = str(majority_vote)

    is_correct = False
    if isinstance(majority_vote, str):
        is_correct = grade(model_answer, majority_vote, fast)
    elif isinstance(majority_vote, list):
        is_correct = False
        for gt in majority_vote:
            is_correct |= grade(model_answer, gt, fast)

    negative_entropy = compute_entropy(counts_copy)
    counts_copy.subtract({model_answer: 1})
    advantage_entropy = negative_entropy - compute_entropy(counts_copy)
    
    if is_correct:
        return {
            "score": advantage_entropy,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": majority_vote,
            "pred": model_answer,
        }
    
    else:
        return {
            "score": advantage_entropy,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": majority_vote,
            "pred": model_answer,
        }

def reward_func_entropy(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = compute_score_entropy(solution_str, ground_truth)

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise

# === Validation score ===

def compute_score_val(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }

def reward_func_val(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = compute_score_val(solution_str, str(ground_truth))

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise


# === Auxiliary Function ===

def compute_SNR(majority_count, runner_up_count, n_total):

    signal_noise_ratio_1 = (majority_count - runner_up_count)**2 / (n_total *  (majority_count + runner_up_count))

    return signal_noise_ratio_1 

def compute_entropy(counts, alpha = 0.5):

    n_total = sum(N for N in counts.values())
    entropy = sum((N+alpha)/(n_total+alpha) * np.log((N+alpha)/(n_total+alpha)) for N in counts.values() if N > 0)

    return entropy