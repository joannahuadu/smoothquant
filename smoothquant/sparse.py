"""
Activation Sparsity Module for SmoothQuant

This module provides utilities to apply N:M activation sparsity to models
using forward hooks, without quantization.

Usage:
    from smoothquant.sparse import apply_activation_sparsity_to_model

    # Apply 2:4 sparsity to model
    apply_activation_sparsity_to_model(model, act_sparsity_n=2, act_sparsity_m=4)
"""

import torch
import torch.nn as nn
from typing import Dict, List


class ActivationSparsityHook:
    """
    Hook manager for applying activation sparsity to linear layers.

    This applies N:M sparsity where M elements are grouped and only N are kept
    (based on magnitude weighted by column importance).
    """

    def __init__(
        self,
        module: nn.Linear,
        act_sparsity_n: int,
        act_sparsity_m: int,
        weight_scoring: bool = True,
    ):
        """
        Initialize the sparsity hook.

        Args:
            module: The Linear module to which this hook is attached
            act_sparsity_n: Number of elements to keep in each group
            act_sparsity_m: Group size (e.g., 2:4 means keep 2 out of every 4)
        """
        self.module = module
        self.act_sparsity_n = act_sparsity_n
        self.act_sparsity_m = act_sparsity_m
        self.weight_scoring = weight_scoring

        # Precompute scale from weight (only once)
        with torch.no_grad():
            weight = module.weight.float()

            if self.weight_scoring:
                # Step 1: Discard weights outside 0.5th-99.5th percentiles
                # Use sampling for large tensors to avoid memory issues
                weight_flat = weight.flatten()
                num_elements = weight_flat.numel()

                # If tensor is too large, use sampling to estimate quantiles
                if num_elements > 1000000:  # Sample if more than 1M elements
                    sample_size = min(100000, num_elements)  # Sample up to 100k elements
                    indices = torch.randperm(num_elements)[:sample_size]
                    weight_sample = weight_flat[indices]
                    q_low = torch.quantile(weight_sample, 0.005)
                    q_high = torch.quantile(weight_sample, 0.995)
                else:
                    q_low = torch.quantile(weight_flat, 0.005)
                    q_high = torch.quantile(weight_flat, 0.995)

                # Create mask for values within the percentile range
                within_range = (weight >= q_low) & (weight <= q_high)

                # Step 2: Standardize the remaining weights using their mean and variance
                if within_range.sum() < 2:
                    # If too few values remain, use original weight
                    weight_processed = weight
                else:
                    # Calculate mean and std from weights within the percentile range
                    w_filtered = weight[within_range]
                    mean = w_filtered.mean()
                    std = w_filtered.std()

                    # Clamp std to avoid division by zero
                    std = std.clamp(min=1e-8)

                    # Standardize all weights using the filtered statistics
                    # Values outside the range are clamped to boundary values
                    weight_processed = (weight - mean) / std
                    weight_processed = weight_processed.clamp(
                        min=(q_low - mean) / std,
                        max=(q_high - mean) / std
                    )

                # Step 3: Calculate column importance based on processed weight norms
                w_col_norm = weight_processed.pow(2).sum(dim=0).sqrt()
            else:
                w_col_norm = weight.pow(2).sum(dim=0).sqrt()
            min_norm = w_col_norm.min().clamp(min=1e-5)
            self.scale = (w_col_norm / min_norm).view(1, -1)

    @torch.no_grad()
    def __call__(self, module, input):
        """
        Forward pre-hook that applies activation sparsity to the input.

        Args:
            module: The module (unused, kept for hook signature)
            input: Tuple containing the input tensor

        Returns:
            Tuple containing the modified input tensor
        """
        x = input[0]
        return (self.apply_activation_sparsity(x),)

    @torch.no_grad()
    def apply_activation_sparsity(self, x):
        """
        Apply N:M activation sparsity to the input tensor.

        The sparsity pattern is determined by:
        1. Grouping features into groups of size M
        2. Within each group, keeping only the N most important activations
        3. Importance is measured by: |activation| * column_importance
        4. Column importance is derived from weight column norms (precomputed in __init__)

        Args:
            x: Input tensor of shape (batch_size, ..., hidden_size)

        Returns:
            Sparsified tensor with the same shape as x
        """
        if self.act_sparsity_n == 0 or self.act_sparsity_m == 0:
            return x

        # Store original shape
        x_shape = x.shape

        # Reshape to 2D: (batch_size * ..., hidden_size)
        x_2d = x.view(-1, x_shape[-1])

        # Compute importance metric for each activation using precomputed scale
        metric = x_2d.abs().float() * self.scale

        # Create sparsity mask
        mask = torch.zeros_like(metric, dtype=torch.bool)

        # Process groups of size M
        for ii in range(0, metric.shape[1], self.act_sparsity_m):
            group_size = min(self.act_sparsity_m, metric.shape[1] - ii)

            # Skip if group is too small
            if group_size < self.act_sparsity_n:
                continue

            # Get the group and find top-k smallest elements to prune
            tmp = metric[:, ii : ii + group_size]
            # Keep the N largest elements (prune the smallest)
            idx = torch.topk(tmp, self.act_sparsity_n, dim=1, largest=False)[1]
            # Mark pruned elements in the mask
            mask.scatter_(1, ii + idx, True)

        # Apply mask: zero out the less important activations
        x_2d = x_2d.masked_fill(mask, 0)

        # Restore original shape
        return x_2d.view(x_shape)


def apply_activation_sparsity_to_model(
    model: nn.Module,
    act_sparsity_n: int = 2,
    act_sparsity_m: int = 4,
    target_modules: List[str] = None,
    weight_scoring: bool = True,
) -> Dict[str, any]:
    """
    Apply activation sparsity hooks to all Linear layers in the model.

    This function registers forward hooks on Linear layers to apply
    N:M activation sparsity during forward pass, without modifying
    the model weights.

    Args:
        model: The PyTorch model to apply sparsity to
        act_sparsity_n: Number of elements to keep in each group (default: 2)
        act_sparsity_m: Group size (default: 4, i.e., 2:4 sparsity)
        target_modules: Optional list of module name patterns to target.
                       If None, all Linear layers are targeted.
        weight_scoring: Whether to apply weight scoring for importance (default: True)

    Returns:
        Dictionary containing hook handles that can be used to remove hooks:
        {
            'hooks': list of hook handles,
            'num_hooks': number of hooks registered
        }

    Example:
        >>> hooks_info = apply_activation_sparsity_to_model(model, 2, 4)
        >>> # Use model with sparsity
        >>> output = model(input)
        >>> # Remove hooks when done
        >>> remove_activation_sparsity_hooks(hooks_info)
    """
    hooks = []

    for name, module in model.named_modules():
        # Skip non-Linear modules
        if not isinstance(module, nn.Linear):
            continue

        # Skip if target_modules is specified and name doesn't match
        if target_modules is not None:
            if any(pattern in name for pattern in target_modules):
                continue

        # Create and register the hook
        hook = ActivationSparsityHook(
            module,
            act_sparsity_n,
            act_sparsity_m,
            weight_scoring=weight_scoring,
        )
        handle = module.register_forward_pre_hook(hook)
        hooks.append(handle)

        print(f"Registered activation sparsity hook ({act_sparsity_n}:{act_sparsity_m}) on: {name}")

    return {
        'hooks': hooks,
        'num_hooks': len(hooks)
    }


def remove_activation_sparsity_hooks(hooks_info: Dict[str, any]):
    """
    Remove activation sparsity hooks from the model.

    Args:
        hooks_info: Dictionary returned by apply_activation_sparsity_to_model
    """
    for handle in hooks_info['hooks']:
        handle.remove()
    print(f"Removed {hooks_info['num_hooks']} activation sparsity hooks")


def get_activation_sparsity_stats(x: torch.Tensor, sparsity_n: int, sparsity_m: int) -> Dict[str, float]:
    """
    Calculate sparsity statistics for a tensor.

    Args:
        x: Input tensor
        sparsity_n: Number of elements to keep in each group
        sparsity_m: Group size

    Returns:
        Dictionary with sparsity statistics
    """
    total_elements = x.numel()
    nonzero_elements = (x != 0).sum().item()
    sparsity = 1.0 - (nonzero_elements / total_elements)

    theoretical_sparsity = 1.0 - (sparsity_n / sparsity_m)

    return {
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'actual_sparsity': sparsity,
        'theoretical_sparsity': theoretical_sparsity,
        'sparsity_ratio': sparsity / theoretical_sparsity if theoretical_sparsity > 0 else 1.0
    }


if __name__ == "__main__":
    # Example usage
    print("Activation Sparsity Module")
    print("=" * 60)
    print("This module provides N:M activation sparsity for PyTorch models.")
    print()
    print("Example:")
    print("  from smoothquant.sparse import apply_activation_sparsity_to_model")
    print("  hooks_info = apply_activation_sparsity_to_model(model, 2, 4)")
    print("  output = model(input)  # Model will have sparse activations")
    print("  remove_activation_sparsity_hooks(hooks_info)  # Clean up")
