import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (MixedPrecisionPolicy,
                                                fully_shard)


def t5_apply_compile(model: nn.Module):
    for name, transformer_block in model.encoder.block.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.encoder.block.register_module(name, transformer_block)


def apply_compile(model: nn.Module, attn_processor):
    """
    Apply torch.compile to each TransformerBlock,
    which makes compilation efficient due to
    repeated structure. Alternatively one can
    compile the whole model (after applying DP).
    """
    for name, transformer_block in model.transformer_blocks.named_children():
        transformer_block.attn.set_processor(attn_processor)
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.transformer_blocks.register_module(name, transformer_block)

    for name, transformer_block in model.single_transformer_blocks.named_children():
        transformer_block.attn.set_processor(attn_processor)
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.single_transformer_blocks.register_module(name, transformer_block)


def t5_apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
):
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in enumerate(model.encoder.block):
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=True,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    depth_single_blocks,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in enumerate(model.transformer_blocks):
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=True,
        )
    for layer_id, transformer_block in enumerate(model.single_transformer_blocks):
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=int(layer_id) < depth_single_blocks - 1,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)
