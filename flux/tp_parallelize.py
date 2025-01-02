import torch
from diffusers.models.transformers.transformer_flux import \
    FluxSingleTransformerBlock
from einops import rearrange
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               PrepareModuleOutput,
                                               RowwiseParallel,
                                               parallelize_module)


def prepare_proj_out_weight(single_block: FluxSingleTransformerBlock, tp_group_size):
    # rowwise
    hidden_dim = 3072
    requires_grad = single_block.proj_out.weight.requires_grad
    linear2_weight_data = single_block.proj_out.weight.data.T.detach().clone()
    out_weight = linear2_weight_data[:hidden_dim, ...]
    out_weight = rearrange(out_weight, "(G D) C -> G D C", G=tp_group_size)
    down_weight = linear2_weight_data.data[hidden_dim:, ...]
    down_weight = rearrange(down_weight, "(G D) C -> G D C", G=tp_group_size)
    new_linear2_weight = torch.cat([out_weight, down_weight], dim=1)
    new_linear2_weight = rearrange(new_linear2_weight, "G D C -> (G D) C")
    single_block.proj_out.weight.data.copy_(new_linear2_weight.T)
    single_block.proj_out.weight.requires_grad_(requires_grad)


def inference_apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    for name, block in model.transformer_blocks.named_children():
        block.attn.heads //= tp_mesh.size()
        layer_plan = {
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "attn.to_out.0": RowwiseParallel(),
            "norm1.linear": ColwiseParallel(output_layouts=Replicate()),
            "ff.net.0.proj": ColwiseParallel(),
            "ff.net.2": RowwiseParallel(),

            "attn.add_q_proj": ColwiseParallel(),
            "attn.add_k_proj": ColwiseParallel(),
            "attn.add_v_proj": ColwiseParallel(),
            "attn.to_add_out": RowwiseParallel(),
            "norm1_context.linear": ColwiseParallel(output_layouts=Replicate()),
            "ff_context.net.0.proj": ColwiseParallel(),
            "ff_context.net.2": RowwiseParallel(),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    for name, block in model.single_transformer_blocks.named_children():
        prepare_proj_out_weight(block, tp_mesh.size())
        block.attn.heads //= tp_mesh.size()
        layer_plan = {
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "proj_mlp": ColwiseParallel(),
            "proj_out": RowwiseParallel(),
            "norm.linear": ColwiseParallel(output_layouts=Replicate()),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    for name, block in model.transformer_blocks.named_children():
        block.attn.heads //= tp_mesh.size()
        layer_plan = {
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "attn.norm_q": PrepareModuleOutput(
                output_layouts=[Shard(1)],
                desired_output_layouts=[Shard(1)],
                use_local_output=True,
            ),
            "attn.norm_k": PrepareModuleOutput(
                output_layouts=[Shard(1)],
                desired_output_layouts=[Shard(1)],
                use_local_output=True,
            ),
            "attn.to_out.0": RowwiseParallel(),
            "norm1.norm": PrepareModuleOutput(
                output_layouts=[Replicate()],
                desired_output_layouts=[Replicate()],
                use_local_output=True,
            ),
            "norm1.linear": ColwiseParallel(
                output_layouts=Replicate(),
            ),
            "ff.net.0.proj": ColwiseParallel(),
            "ff.net.2": RowwiseParallel(),

            "attn.add_q_proj": ColwiseParallel(),
            "attn.add_k_proj": ColwiseParallel(),
            "attn.norm_added_q": PrepareModuleOutput(
                output_layouts=[Shard(1)],
                desired_output_layouts=[Shard(1)],
                use_local_output=True,
            ),
            "attn.norm_added_k": PrepareModuleOutput(
                output_layouts=[Shard(1)],
                desired_output_layouts=[Shard(1)],
                use_local_output=True,
            ),
            "attn.add_v_proj": ColwiseParallel(),
            "attn.to_add_out": RowwiseParallel(),
            "norm1_context.norm": PrepareModuleOutput(
                output_layouts=[Replicate()],
                desired_output_layouts=[Replicate()],
                use_local_output=True,
            ),
            "norm1_context.linear": ColwiseParallel(
                output_layouts=Replicate(),
            ),
            "ff_context.net.0.proj": ColwiseParallel(),
            "ff_context.net.2": RowwiseParallel(),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
        auxiliary_plan = {
            "attn.norm_q": PrepareModuleInput(
                input_layouts=[Shard(1)],
                desired_input_layouts=[Shard(1)],
            ),
            "attn.norm_k": PrepareModuleInput(
                input_layouts=[Shard(1)],
                desired_input_layouts=[Shard(1)],
            ),
            "attn.norm_added_q": PrepareModuleInput(
                input_layouts=[Shard(1)],
                desired_input_layouts=[Shard(1)],
            ),
            "attn.norm_added_k": PrepareModuleInput(
                input_layouts=[Shard(1)],
                desired_input_layouts=[Shard(1)],
            ),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=auxiliary_plan,
        )

    for name, block in model.single_transformer_blocks.named_children():
        prepare_proj_out_weight(block, tp_mesh.size())
        block.attn.heads //= tp_mesh.size()
        layer_plan = {
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "attn.norm_q": PrepareModuleOutput(
                output_layouts=[Shard(1)],
                desired_output_layouts=[Shard(1)],
                use_local_output=True,
            ),
            "attn.norm_k": PrepareModuleOutput(
                output_layouts=[Shard(1)],
                desired_output_layouts=[Shard(1)],
                use_local_output=True,
            ),
            "proj_mlp": ColwiseParallel(),
            "proj_out": RowwiseParallel(),
            "norm.linear": ColwiseParallel(output_layouts=Replicate()),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
        auxiliary_plan = {
            "attn.norm_q": PrepareModuleInput(
                input_layouts=[Shard(1)],
                desired_input_layouts=[Shard(1)],
            ),
            "attn.norm_k": PrepareModuleInput(
                input_layouts=[Shard(1)],
                desired_input_layouts=[Shard(1)],
            ),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=auxiliary_plan,
        )

    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan={
            "x_embedder": PrepareModuleInput(
                input_layouts=[Replicate()],
                desired_input_layouts=[Replicate()],
            ),
            "context_embedder": PrepareModuleInput(
                input_layouts=[Replicate()],
                desired_input_layouts=[Replicate()],
            ),
            "time_text_embed": PrepareModuleInput(
                input_layouts=[Replicate(), Replicate(), Replicate()],
                desired_input_layouts=[Replicate(), Replicate(), Replicate()],
            ),
            "time_text_embed.time_proj": PrepareModuleInput(
                input_layouts=[Replicate()],
                desired_input_layouts=[Replicate()],
                use_local_output=True,
            ),
            "time_text_embed.timestep_embedder": PrepareModuleInput(
                input_layouts=[Replicate()],
                desired_input_layouts=[Replicate()],
            ),
            "time_text_embed.guidance_embedder": PrepareModuleInput(
                input_layouts=[Replicate()],
                desired_input_layouts=[Replicate()],
            ),
            "norm_out": PrepareModuleInput(
                input_layouts=[Replicate(), Replicate()],
                desired_input_layouts=[Replicate(), Replicate()],
            ),
        },
    )
    auxiliary_plan = {
        "x_embedder": PrepareModuleOutput(
            output_layouts=[Replicate()],
            desired_output_layouts=[Replicate()],
            use_local_output=True,
        ),
        "context_embedder": PrepareModuleOutput(
            output_layouts=[Replicate()],
            desired_output_layouts=[Replicate()],
            use_local_output=True,
        ),
        "proj_out": PrepareModuleOutput(
            output_layouts=[Replicate()],
            desired_output_layouts=[Replicate()],
            use_local_output=True,
        ),
    }
    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=auxiliary_plan,
    )
    distribute_module(
        module=model,
        device_mesh=tp_mesh,
    )
