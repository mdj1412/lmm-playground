import re

import torch
from torch import nn
import torch.nn.functional as F


IGNORE_INDEX = -100


def get_lprobs(logits, target, attention_mask):
    """
    Get log probabilities from logits and target
    """
    lprobs = logits[..., :-1, :].log_softmax(-1)
    labels = target[..., 1:].clone()
    mask = attention_mask[..., 1:].to(lprobs.dtype)
    labels[labels < 0] = 0  # should be masked out anyway
    lprobs = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # masked mean
    lprobs = (lprobs * mask).sum(-1) / mask.sum(-1)
    return lprobs


def get_sft_loss(logits, labels, attention_mask):
    """
    Get softmax loss
    """
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:]
        shift_logits = logits[..., :-1, :][
            shift_attention_mask.to(logits.device) != 0
        ].contiguous()
        shift_labels = labels[..., 1:][
            shift_attention_mask.to(labels.device) != 0
        ].contiguous()
    else:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1).to(shift_logits.device),
    )
    return loss


def is_number_regex(input_str):
    return bool(re.match(r"^[0-9]+$", input_str))


def l2_dist(x, y):
    B, L = x.shape
    # shape: x (BL) y (V)
    dist = (x.reshape(-1)[:, None] - y[None, :]) ** 2
    dist = dist.reshape(B, L, -1)
    return dist


def get_digit_loss(
    logits,
    labels,
    attention_mask=None,
    tokenizer=None,
    target_temperature=4.0,
    beta: float = 1.0,
):
    sft_loss = get_sft_loss(logits, labels, attention_mask)

    # shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    digits = [(k, v) for k, v in tokenizer.vocab.items() if is_number_regex(k)]
    digit_values = (
        torch.tensor([int(k) for k, v in digits])
        .to(shift_labels.device)
        .to(logits.dtype)
    )
    digit_indices = torch.tensor([v for k, v in digits]).to(shift_labels.device)

    mask = shift_labels[..., None] == digit_indices[None, None, :]
    any_mask = mask.any(-1).flip(-1)

    x = any_mask.long()
    # Compute the cumulative sum
    cumsum = torch.cumsum(x, dim=-1)

    # Create a mask to identify positions where cumulative sum should reset
    reset_mask = x == 0

    # Compute the differences in the cumulative sum at reset points
    reset_diff = torch.zeros_like(cumsum)
    reset_diff[:, 1:] = cumsum[:, :-1] * reset_mask[:, 1:].to(dtype=cumsum.dtype)
    # Compute the cumulative sum again after reset
    adjusted_cumsum = cumsum - reset_diff.cummax(-1).values
    pos = adjusted_cumsum.flip(-1)

    data_targets = mask.long().argmax(-1)
    data_targets = digit_values[data_targets.reshape(-1)].reshape(*data_targets.shape)
    digit_targets = l2_dist(data_targets, digit_values)
    digit_targets = (-digit_targets / target_temperature).softmax(-1)

    digit_logits = shift_logits[..., digit_indices]
    loss_fn = nn.KLDivLoss(reduction="none")
    loss = loss_fn(digit_logits.log_softmax(-1), digit_targets)  # BLV
    loss = loss.sum(-1)  # definition of KL

    loss_coeff = pos
    loss_coeff = loss_coeff > 0
    loss_coeff = loss_coeff.to(loss.dtype)
    loss = (loss * loss_coeff).sum() / loss_coeff.sum()
    return beta * loss + sft_loss


def get_digit_base_loss(
    logits,
    labels,
    attention_mask=None,
    tokenizer=None,
    beta: float = 1.0,
):
    sft_loss = get_sft_loss(logits, labels, attention_mask)

    # shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    digits = [(k, v) for k, v in tokenizer.vocab.items() if is_number_regex(k)]
    digit_indices = torch.tensor([v for k, v in digits]).to(shift_labels.device)

    shift_labels = shift_labels.reshape(-1)
    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))

    mask = shift_labels[..., None] == digit_indices[None, :]
    data_targets = mask.long().argmax(-1)
    any_mask = mask.any(-1)

    digit_logits = shift_logits[..., digit_indices]
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(digit_logits, data_targets)  # BLV
    mask = any_mask.to(loss.dtype)
    loss = (loss * mask).sum() / mask.sum()
    return beta * loss + sft_loss
