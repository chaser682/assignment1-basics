from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import math
from typing import Dict
import torch.nn.functional as F

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    out_features = F.linear(in_features, weights)
    return out_features

    # raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embeddings = weights[token_ids]
    return embeddings

    # raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    
    r'''
    swiglu公式：
      $FFN(x) = (\text{SiLU}(x W_1) \odot (x W_3)) W_2$
    '''
    # 计算门
    gate_proj = F.linear(in_features, w1_weight)
    # 应用silu激活函数
    gate = F.silu(gate_proj)
    # 计算上采样
    up_proj = F.linear(in_features, w3_weight)
    # 逐元素相乘
    gated_output = gate * up_proj
    # 计算下采样
    out_features = F.linear(gated_output, w2_weight)
    return out_features

    # raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    score = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    attention = run_softmax(score / math.sqrt(Q.size(-1)), dim=-1) 
    output = torch.matmul(attention, V)
    return output

    # raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    assert d_k % num_heads == 0, "d_k 必须能被 num_heads 整除"
    assert d_v % num_heads == 0, "d_v 必须能被 num_heads 整除"
    d_head_k = d_k // num_heads
    d_head_v = d_v // num_heads
    batch_shape = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    q = F.linear(in_features, q_proj_weight).view(-1, seq_len, num_heads, d_head_k).transpose(-3, -2)
    k = F.linear(in_features, k_proj_weight).view(-1, seq_len, num_heads, d_head_k).transpose(-3, -2)
    v = F.linear(in_features, v_proj_weight).view(-1, seq_len, num_heads, d_head_v).transpose(-3, -2)
    # 创建因果掩码，下三角矩阵
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool), 
        diagonal=0
    )
    out = run_scaled_dot_product_attention(q, k, v, causal_mask)
    out = out.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, d_v)
    out_features = F.linear(out, o_proj_weight)
    return out_features

    # raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    assert d_k % num_heads == 0, "d_k 必须能被 num_heads 整除"
    assert d_v % num_heads == 0, "d_v 必须能被 num_heads 整除"
    d_head_k = d_k // num_heads
    d_head_v = d_v // num_heads
    batch_shape = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    q = F.linear(in_features, q_proj_weight).view(-1, seq_len, num_heads, d_head_k).transpose(-3, -2)
    k = F.linear(in_features, k_proj_weight).view(-1, seq_len, num_heads, d_head_k).transpose(-3, -2)
    v = F.linear(in_features, v_proj_weight).view(-1, seq_len, num_heads, d_head_v).transpose(-3, -2)
    # 对q和k应用RoPE
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)
    q = run_rope(d_head_k, theta, max_seq_len, q, token_positions)
    k = run_rope(d_head_k, theta, max_seq_len, k, token_positions)
    # 创建因果掩码，下三角矩阵
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool), 
        diagonal=0
    )
    out = run_scaled_dot_product_attention(q, k, v, causal_mask)
    out = out.transpose(-3, -2).contiguous().view(*batch_shape, seq_len, d_v)
    out_features = F.linear(out, o_proj_weight)
    return out_features

    # raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    assert d_k % 2 == 0, "d_k 必须是偶数"
    d_half = d_k // 2
    dev = in_query_or_key.device

    # 计算 RoPE 的频率
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float, device=dev) / d_k))
    # 计算位置编码
    position = torch.arange(0, max_seq_len, dtype=torch.float, device=dev)
    # 使用 einsum 或 outer 计算 m * freq_i 的角度矩阵
    angles = torch.einsum("i, j -> ij", position, inv_freq) # 等价于 position.unsqueeze(1) * inv_freq.unsqueeze(0)
    # 查找当前 token 位置的角度
    gathered_angles = angles[token_positions]
    # 计算 sin 和 cos
    sin_angles = torch.sin(gathered_angles)
    cos_angles = torch.cos(gathered_angles)
    # 应用旋转
    x_even = in_query_or_key[..., ::2] # [x0, x2, ...]
    x_odd = in_query_or_key[..., 1::2] # [x1, x3, ...]
    out_even = x_even * cos_angles - x_odd * sin_angles
    out_odd  = x_even * sin_angles + x_odd * cos_angles
    # 创建一个空张量来重新交错结果
    x_rotated = torch.empty_like(in_query_or_key)
    # 将旋转后的值放回
    x_rotated[..., ::2] = out_even # [x0', x2', ...]
    x_rotated[..., 1::2] = out_odd  # [x1', x3', ...]

    return x_rotated.type_as(in_query_or_key)


    # raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    x = in_features
    x = x + run_multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights['attn.q_proj.weight'],
        weights['attn.k_proj.weight'],
        weights['attn.v_proj.weight'],
        weights['attn.output_proj.weight'],
        run_rmsnorm(
            d_model,
            eps=1e-5, # 注意eps默认值应该为1e-5
            weights=weights['ln1.weight'],
            in_features=x
        )
    )
    x = x + run_swiglu(
        d_model,
        d_ff,
        weights['ffn.w1.weight'],
        weights['ffn.w2.weight'],
        weights['ffn.w3.weight'],
        run_rmsnorm(
            d_model,
            eps=1e-5,
            weights=weights['ln2.weight'],
            in_features=x
        )
    )
    return x

    # raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    r"""Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    x = run_embedding(
        vocab_size,
        d_model,
        weights['token_embeddings.weight'],
        in_indices
    )
    for i in range(num_layers):
        x = run_transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            {
                'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
                'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
                'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
                'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
                'ln1.weight': weights[f'layers.{i}.ln1.weight'],
                'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
                'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight'],
                'ffn.w3.weight': weights[f'layers.{i}.ffn.w3.weight'],
                'ln2.weight': weights[f'layers.{i}.ln2.weight'],
            },
            x
        )
    x = run_rmsnorm(
        d_model,
        eps=1e-5,
        weights=weights['ln_final.weight'],
        in_features=x
    )
    x = F.linear(x, weights['lm_head.weight'])
    # 注意这里不需要softmax，交叉熵损失函数里会包含softmax
    # x = run_softmax(x, dim=-1) 
    return x

    # raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    denom = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    output = in_features / denom * weights
    return output

    # raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    '''
    silu公式：
        SiLU(x) = x × σ(x)
        其中，σ(x) = 1 / (1 + e^(-x))
    '''
    sigmoid_out = 1.0 / (1.0 + torch.exp(-1.0 * in_features))
    output = in_features * sigmoid_out
    return output

    # raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    dataset = torch.from_numpy(dataset).to(device).long()
    start_idx = torch.randint(0, dataset.shape[0] - context_length, (batch_size,), device=device)
    seq_ids = dataset[start_idx.unsqueeze(1) + torch.arange(context_length + 1, device=device).unsqueeze(0)]
    inputs = dataset[seq_ids[:, :-1]]
    labels = dataset[seq_ids[:, 1:]]

    return inputs, labels

    # raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    # 提升数值的稳定性，再exp之前减去最大值
    shifted = in_features - in_features.max(dim=dim, keepdim=True).values
    # 计算softmax
    exp_values = torch.exp(shifted)
    softmax_values = exp_values / exp_values.sum(dim=dim, keepdim=True)
    return softmax_values
    # raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 计算softmax
    # 数值稳定性处理，每行qu减去最大值
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    # 计算softmax的分子和分母
    exp_values = torch.exp(shifted)
    partition = exp_values.sum(dim=-1, keepdim=True)
    # 计算log softmax
    log_probs = shifted - torch.log(partition)
    # 计算交叉熵损失
    ce_loss = -log_probs[torch.arange(inputs.size(0)), targets]
    return ce_loss.mean(dim=-1)
    # raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # 计算所有梯度的总l2范数
    l2_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    # 如果总l2范数超过max_l2_norm，则进行裁剪
    if l2_norm > max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + 1e-12) # 避免除以零
        for g in grads:
            g.mul_(clip_coef)
    # raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    class AdamW(torch.nn.Module):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            super().__init__()
            self.lr = lr
            self.betas = betas
            self.eps = eps
            self.weight_decay = weight_decay
            # 将参数存储为列表
            self.params = list(params)
            # state 现在将以参数对象本身为键
            self.state = {}
            # 创建一个从参数对象到其索引的映射，用于 state_dict
            self.param_to_idx = {p: i for i, p in enumerate(self.params)}
            # 创建一个从索引到参数对象的映射，用于 load_state_dict
            self.idx_to_param = {i: p for p, i in self.param_to_idx.items()}


        def zero_grad(self):
            for p in self.params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

        # 优化了 step 方法：
        # 1. 使用正确的 AdamW (解耦权重衰减) 逻辑。
        # 2. 对动量和方差使用 in-place (原地) 更新，以提高效率。
        def step(self):
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                # 1. AdamW: 解耦权重衰减
                # 在计算梯度更新之前，直接将权重衰减应用到参数上
                # p.data = p.data - lr * wd * p.data
                if self.weight_decay != 0.0:
                    p.data.mul_(1.0 - self.lr * self.weight_decay)

                # 初始化状态
                if p not in self.state:
                    self.state[p] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(p.data),
                        'exp_avg_sq': torch.zeros_like(p.data)
                    }
                
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = self.betas
                
                # 更新步数
                state['step'] += 1
                
                # 2. 更新一阶矩 (动量) - In-place
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 3. 更新二阶矩 (方差) - In-place
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj() if grad.is_complex() else grad, value=1 - beta2)
                
                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                # Adam 更新的分母
                denom = torch.sqrt(exp_avg_sq_hat).add_(self.eps)
                
                # 4. In-place 参数更新 (Adam 部分)
                # p.data = p.data - self.lr * (exp_avg_hat / denom)
                p.data.addcdiv_(exp_avg_hat, denom, value=-self.lr)

                # 5. 原始错误的权重衰减位置
                # p.data = p.data - self.lr * self.weight_decay * p.data

        # state_dict 使用索引 (index) 而不是 id(p)
        def state_dict(self) -> Dict[str, Any]:
            """
            返回与 torch.optim.Optimizer.state_dict() 兼容的结构。
            使用参数索引 (index) 作为 'state' 的键和 'param_groups'['params'] 的条目。
            """
            # state: 将 self.state (以 param obj 为键) 转换为以 index 为键
            state_out = {}
            for p, s in self.state.items():
                pidx = self.param_to_idx.get(p)
                if pidx is None:
                    continue # 参数已不在优化器中 (罕见)
                    
                state_entry = {}
                for k, v in s.items():
                    state_entry[k] = v.clone() if torch.is_tensor(v) else v
                state_out[pidx] = state_entry # 使用索引作为键

            # param_groups: 使用索引列表
            param_group = {
                'params': list(self.idx_to_param.keys()), # 即 [0, 1, 2, ...]
                'lr': self.lr,
                'betas': self.betas,
                'eps': self.eps,
                'weight_decay': self.weight_decay,
            }

            return {'state': state_out, 'param_groups': [param_group]}
        
        # load_state_dict 现在索引 (index) 加载
        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            """
            将来自 state_dict 的状态加载回 optimizer。
            state_dict 的 'state' 键是以 param_index 为键的映射；
            我们需要把它转换为以 param obj 为键的 self.state。
            """
            if not isinstance(state_dict, dict):
                raise TypeError("state_dict must be a dict")

            state_in = state_dict.get('state', {})
            param_groups_in = state_dict.get('param_groups', [])
            
            if not param_groups_in:
                raise ValueError("state_dict 缺少 'param_groups'，无法将旧ID映射到新参数。")

            # 获取保存的索引列表
            old_param_indices = param_groups_in[0]['params']
            
            # 检查参数数量是否匹配
            if len(old_param_indices) != len(self.params):
                raise ValueError(
                    f"参数列表长度不匹配：已保存 {len(old_param_indices)} vs 当前 {len(self.params)}"
                )
            
            # (我们假设 self.params 的当前顺序与保存时的索引一致)
            
            # 使用索引来恢复状态
            new_state = {}
            for idx_str, s in state_in.items():
                # state_in 中的 key 应该是索引（可能为字符串）
                idx = int(idx_str)
                
                # 查找此索引对应的新参数对象
                p = self.idx_to_param.get(idx)
                
                if p is None:
                    # state_dict 中的索引超出了当前优化器的参数范围
                    continue

                # 恢复每个子项；如果是 tensor，确保它在新参数的 device 上
                state_entry = {}
                for k, v in s.items():
                    if torch.is_tensor(v):
                        # 将 tensor 放到参数的 device
                        state_entry[k] = v.to(p.device).clone()
                    else:
                        state_entry[k] = v
                new_state[p] = state_entry # 内部 self.state 仍然使用 param obj 作为键

            self.state = new_state

            # 恢复 param_groups 的超参数 (这部分原先就是正确的)
            if len(param_groups_in) > 0:
                pg0 = param_groups_in[0]
                if 'lr' in pg0:
                    self.lr = pg0['lr']
                if 'betas' in pg0:
                    self.betas = tuple(pg0['betas'])
                if 'eps' in pg0:
                    self.eps = pg0['eps']
                if 'weight_decay' in pg0:
                    self.weight_decay = pg0['weight_decay']

    return AdamW

    # raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # 线性预热
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    # 计算余弦退火学习率
    # 注意：cosine_cycle_iters指的是余弦退火在整个训练过程中的迭代次数，而不是从0开始的迭代次数
    elif it < cosine_cycle_iters:
        # 计算余弦退火阶段进度
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 余弦退火公式：lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    # 学习率保持在最小值
    else:
        return min_learning_rate
    
    # raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

    # raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration

    # raise NotImplementedError


import re
import regex
from typing import Any, Optional, List, Dict, Tuple, Set
from typing import Iterable, Iterator

# gpt2 分割正则表达式
GPT2_WORD_SPLIT_PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# BPE Tokenizer class
class BPETokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        # 设置 gpt2 的 word 分割器
        self.word_split_re = GPT2_WORD_SPLIT_PATTERN
        # 设置特殊Token
        self.special_tokens: Set[str] = set(special_tokens or [])
        # 特殊token的正则表达式模式列表
        special_token_patterns = []
        if special_tokens is not None:
            # 注意，为了防止特殊token之间重叠问题，应该按照长度从长到短排序
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in special_tokens:
                # 为特殊token创建正则表达式模式
                special_token_patterns.append(re.escape(special_token))
                # 转换为bytes形式
                special_token_bytes = special_token.encode('utf-8')
                # 将 special token 添加到词汇表中
                if special_token_bytes not in set(vocab.values()):
                    vocab[len(vocab)] = special_token_bytes
                
        
        # 生成分割特殊token的正则表达式，用来分割输入字符串
        if special_token_patterns:
            self.special_tokens_re = re.compile(f"({'|'.join(special_token_patterns)})")
        else:
            self.special_tokens_re = None

        # 设置解码器和编码器(id -> token bytes 和 token bytes -> id)
        self.decoder: Dict[int, bytes] = vocab
        self.encoder: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        
        # 设置BPE合并规则
        # 简单将顺序作为合并优先级（越早的对优先级越高）
        self.merges: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # 合并缓存，用于加速已处理过的文本块
        self.cache: Dict[bytes, List[bytes]] = {}

        # 缓存基础字节的编码器，用于快速回退
        self.byte_encoder: Dict[bytes, int] = {
            bytes([i]): self.encoder[bytes([i])] 
            for i in range(256) if bytes([i]) in self.encoder
        }
    
    # 从一个token列表中获取所有相邻的字节对。
    def _get_pairs(self, tokens: List[bytes]) -> Set[Tuple[bytes, bytes]]:
        if len(tokens) < 2:
            return set()
        return set(zip(tokens, tokens[1:]))

    # 对单个字节块应用BPE合并规则
    def _merge_chunk(self, chunk_bytes: bytes) -> List[bytes]:
        # 检测缓存
        if chunk_bytes in self.cache:
            return self.cache[chunk_bytes]
        
        # 初始时，token列表是单个字节的列表
        tokens: List[bytes] = [bytes([b]) for b in chunk_bytes]

        while True:
            # 获取所有相邻的对
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            # 找到优先级最高的对合并
            best_pair = min(
                pairs,
                key=lambda pair: self.merges.get(pair, float('inf'))
            )
            # 如果该对不在合并规则中，停止
            if best_pair not in self.merges:
                break
            # 执行合并，构建新的token列表
            new_token = best_pair[0] + best_pair[1]
            new_tokens: List[bytes] = []
            i = 0
            # 查找要合并的对
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2 # 跳过两个已合并的token
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens # 更新token列表
        
        # 缓存结果
        self.cache[chunk_bytes] = tokens
        return tokens
    
    # 将输入的字符串编码为 token ID 列表
    def encode(self, text: str) -> List[int]:
        # 根据特殊token分割字符串
        if self.special_tokens_re:
            chunks = self.special_tokens_re.split(text)
        else:
            chunks = [text]
        
        token_ids: List[int] = []
        for chunk in chunks:
            # 空文本块跳过
            if not chunk:
                continue
            # 如果是特殊token，直接encode
            if chunk in self.special_tokens:
                token_ids.append(self.encoder[chunk.encode('utf-8')])
            # 否则，引用BPE编码
            else:
                # 使用 gpt2 regex 将其预分割为 words
                for word_chunk in self.word_split_re.findall(chunk):
                    # 对每个 word chunk 应用 BPE 合并
                    merged_tokens = self._merge_chunk(word_chunk.encode('utf-8'))
                    # 将合并后的token bytes转换为token IDs
                    for token in merged_tokens:
                        if token in self.encoder:
                            token_ids.append(self.encoder[token])
                        else: # 回退到字节级编码
                            token_ids.extend(self.byte_encoder[bytes([b])] for b in token if bytes([b]) in self.byte_encoder)
            
        return token_ids
    
    # 将一个字符串的可迭代对象编码为 token ID 的迭代器
    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        # 遍历可迭代对象（例如，一次从文件中读取一行）
        for text in texts:
            # 使用 'yield from' 立即返回当前行
            #    的 token ID，而不是将它们附加到列表中。
            #    这会将 [10, 20, 30] 这样的列表 "展开" 为 
            #    yield 10, then yield 20, then yield 30
            yield from self.encode(text)
    
    # 将 token ID 列表解码回字符串
    def decode(self, token_ids: List[int]) -> str:
        # 将所有 token IDs 转换为字节，并连接起来
        all_bytes = b''.join(self.decoder[token_id] for token_id in token_ids if token_id in self.decoder)

        # 将完整的字节序列一次性解码为字符串，使用 errors='replace' 来处理无效的UTF-8序列（例如被截断的多字节字符）
        return all_bytes.decode('utf-8', errors='replace')


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)

    # raise NotImplementedError

from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 计算 word 频率计数器中所有相邻字节对的频率。
def _get_stats(word_freqs: Counter) -> Counter:
    pair_counts = Counter()
    for word_tokens, freq in word_freqs.items():
        # 遍历一个 word 中的所有相邻对
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i+1])
            pair_counts[pair] += freq
    return pair_counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 初始化词汇表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 特殊token正则表达式模式列表
    special_token_patterns = []
    # 添加特殊tokens
    for special_token in special_tokens:
        special_token_patterns.append(re.escape(special_token))
        special_token_bytes = special_token.encode('utf-8')
        if special_token_bytes not in set(vocab.values()):
            vocab[len(vocab)] = special_token_bytes
    special_token_set = set(special_tokens)
    # 计算特殊token正则表达式分割模式
    if special_token_patterns:
        special_tokens_re = re.compile(f"({'|'.join(special_token_patterns)})")
    else:
        special_tokens_re = None
    
    # 根据词汇表计算合并次数
    num_merges = vocab_size - len(vocab)
    merges: List[Tuple[bytes, bytes]] = []
    if num_merges < 0:
        return vocab, merges

    # 预分词并统计word频率
    word_freqs = Counter()
    try:
        with open(input_path, 'rb') as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                file_chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # 先使用特殊token分割行
                if special_tokens_re:
                    chunks = special_tokens_re.split(file_chunk)
                else:
                    chunks = [file_chunk]
                for chunk in chunks:
                    if not chunk:
                        continue
                    # 如果是特殊token，不进行计数
                    if chunk in special_token_set:
                        continue
                    else:
                        # 否则使用 gpt2 regex 分割器 进行分割
                        words = GPT2_WORD_SPLIT_PATTERN.findall(chunk)
                        for word in words:
                            word_bytes = word.encode('utf-8')
                            # 初始时，一个 word 是其构成字节的元组
                            tokens = tuple(bytes([b]) for b in word_bytes)
                            word_freqs[tokens] += 1
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return {}, [] # 返回空
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return {}, [] # 返回空
    
    # 只需计算初始的 pair_counts，每次word频率更新时增量更新
    pair_counts = _get_stats(word_freqs)
    # 训练循环
    for i in range(num_merges):
        if not pair_counts:
            break

        # 找到频率最高且字典序最大的对进行合并
        max_freq = max(pair_counts.values())
        candidates = [
            pair for pair, freq in pair_counts.items() 
            if freq == max_freq
        ]
        best_pair = max(candidates) # 字典序最大的对
        merges.append(best_pair)

        # 添加到词汇表
        new_token = best_pair[0] + best_pair[1]
        if new_token not in set(vocab.values()):
            vocab[len(vocab)] = new_token

        # 合并字节对
        # 更新 word_freqs 并同时增量更新 pair_counts
        new_word_freqs = Counter()
        for word_tokens, freq in word_freqs.items():
            new_tokens = []
            idx = 0
            has_changed = False # 跟踪这个词是否被修改了

            while idx < len(word_tokens):
                # 检查是否是我们要合并的对
                if idx < len(word_tokens) - 1 and (word_tokens[idx], word_tokens[idx+1]) == best_pair:
                    has_changed = True
                    # --- 1. 减去旧的字节对频率 ---
                    # 减去 左侧 受影响的对 (token[idx-1], token[idx])
                    if idx > 0:
                        pair = (word_tokens[idx-1], word_tokens[idx])
                        pair_counts[pair] -= freq
                        if pair_counts[pair] == 0: pair_counts.pop(pair)
                    # 减去 右侧 受影响的对 (token[idx+1], token[idx+2])
                    if idx < len(word_tokens) - 2:
                        pair = (word_tokens[idx+1], word_tokens[idx+2])
                        pair_counts[pair] -= freq
                        if pair_counts[pair] == 0: pair_counts.pop(pair)
                    # 减去被合并的对 best_pair: token[idx], token[idx+1])
                    pair_counts[best_pair] -= freq
                    if pair_counts[best_pair] == 0: pair_counts.pop(best_pair)
                    
                    # --- 2. 添加新 token 并 加上新的字节对频率 ---
                    new_tokens.append(new_token)
                    # 加上 左侧 新生成的对 (new_tokens[-2], new_token)
                    if len(new_tokens) >= 2: # 相当于 if idx > 0
                        pair = (new_tokens[-2], new_token)
                        pair_counts[pair] += freq
                    # 加上 右侧 新生成的对 (new_token, token[idx+2])
                    if idx < len(word_tokens) - 2:
                        pair = (new_token, word_tokens[idx+2])
                        pair_counts[pair] += freq
                    idx += 2 # 跳过两个已合并的 token
                else:
                    # --- 3. 没有合并，正常处理 ---
                    current_token = word_tokens[idx]
                    new_tokens.append(current_token)
                    idx += 1
            
            # 循环结束后，将新的word添加到新的word_freqs中
            if has_changed:
                new_word_freqs[tuple(new_tokens)] += freq
            else:
                # 如果这个词没有发生任何合并，直接使用原key
                new_word_freqs[word_tokens] += freq
        
        word_freqs = new_word_freqs
        # if (i + 1) % 50 == 0 or i == num_merges - 1:
        #     print(f"  Merge {i+1}/{num_merges}: Merged {best_pair} -> {new_token}. Vocab size: {len(vocab)}")

    return vocab, merges

    # raise NotImplementedError
