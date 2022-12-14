# coding=utf-8



import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        # Copy so the input remains unchanged.
        logits = vocab_parallel_logits.clone()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)


class _ParallelKLLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        targets_max = torch.max(targets, dim=-1)[0]
        torch.distributed.all_reduce(targets_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        targets.sub_(targets_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_targets = targets.exp()
        sum_exp_targets = exp_targets.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_targets,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # targets_softmax: [b, s, v_p]
        targets_softmax = torch.div(exp_targets, sum_exp_targets.unsqueeze(-1))

        # sum_targets_softmax_logits: [b, s]
        sum_targets_softmax_logits = torch.matmul(
            targets_softmax.unsqueeze(-2), logits.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        torch.distributed.all_reduce(sum_targets_softmax_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        log_targets_softmax = torch.log(targets_softmax)
        sum_log_targets_softmax = torch.matmul(
            targets_softmax.unsqueeze(-2), log_targets_softmax.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        torch.distributed.all_reduce(sum_log_targets_softmax,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())


        loss = torch.log(sum_exp_logits) - sum_targets_softmax_logits + sum_log_targets_softmax

        logits_softmax = torch.div(exp_logits, sum_exp_logits.unsqueeze(-1))

        cls.save_for_backward(logits_softmax, targets_softmax)

        return loss

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        logits_softmax, targets_softmax = cls.saved_tensors
        grad_input = (logits_softmax - targets_softmax) * grad_output.unsqueeze(-1)

        return grad_input, None


def parallel_KL_loss(logits, targets):
    return _ParallelKLLoss.apply(logits, targets)


class _ParallelSoftCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(cls, logits: torch.Tensor, targets: torch.Tensor):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # sum_targets_softmax_logits: [b, s]
        sum_targets_softmax_logits = torch.matmul(
            targets.unsqueeze(-2), logits.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        torch.distributed.all_reduce(sum_targets_softmax_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        loss = torch.log(sum_exp_logits) - sum_targets_softmax_logits

        logits_softmax = torch.div(exp_logits, sum_exp_logits.unsqueeze(-1))

        cls.save_for_backward(logits_softmax, targets)

        return loss

    @staticmethod
    def backward(cls, grad_output: torch.Tensor):
        logits_softmax, targets = cls.saved_tensors
        grad_input = (logits_softmax - targets) * grad_output.unsqueeze(-1)

        return grad_input, None


def parallel_soft_cross_entropy_loss(logits, targets):
    return _ParallelSoftCrossEntropyLoss.apply(logits, targets)