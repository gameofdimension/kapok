import random
import sys

import torch

try:
    import torch_npu  # type: ignore # noqa
    from torch_npu.contrib import transfer_to_npu  # type: ignore # noqa
    is_npu = True
except ImportError:
    print("torch_npu not found")
    is_npu = False
import torch.distributed as dist
from diffusers import FluxPipeline
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from torch.distributed.device_mesh import init_device_mesh

from flux.data import make_dataloader
from flux.fsdp_parallelize import (apply_compile, apply_fsdp, t5_apply_compile,
                                   t5_apply_fsdp)
from flux.tp_parallelize import apply_tp, t5_apply_tp
from tool.utils import cleanup, init_distributed


def make_infer_pipeline(dist_type, device):
    dtype = torch.bfloat16
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
    )

    if not is_npu:
        mesh = init_device_mesh(
            "cuda", (dist.get_world_size(),))
        if dist_type == 'fsdp':
            apply_compile(pipeline.transformer, FluxAttnProcessor2_0())
            apply_fsdp(pipeline.transformer, mesh, pipeline.transformer.config.num_single_layers)
            t5_apply_compile(pipeline.text_encoder_2)
            t5_apply_fsdp(pipeline.text_encoder_2, mesh)
        else:
            apply_tp(pipeline.transformer, mesh)
            apply_compile(pipeline.transformer, FluxAttnProcessor2_0())
            t5_apply_tp(pipeline.text_encoder_2, mesh)
            t5_apply_compile(pipeline.text_encoder_2)
    else:
        mesh = init_device_mesh(
            "npu", (dist.get_world_size(),))
        if dist_type == 'fsdp':
            apply_fsdp(pipeline.transformer, mesh, pipeline.transformer.config.num_single_layers)
            t5_apply_fsdp(pipeline.text_encoder_2, mesh)
        else:
            apply_tp(pipeline.transformer, mesh)
            t5_apply_tp(pipeline.text_encoder_2, mesh)

    pipeline = pipeline.to(device=device, dtype=dtype)

    torch.cuda.empty_cache()
    return pipeline


def main():
    seed = int(sys.argv[1])
    dist_type = sys.argv[2]
    round = int(sys.argv[3])
    assert dist_type in ['fsdp', 'tp']
    init_distributed()

    random.seed(seed)
    pipeline = make_infer_pipeline(dist_type, device='cuda')
    generator = torch.Generator("cpu").manual_seed(seed)

    rank = dist.get_rank()
    height, width = 1024, 1024
    batch_size = 1
    path = './flux/prompts/PartiPrompts.tsv'
    dataloader = make_dataloader(path=path, batch_size=batch_size)

    for batch in dataloader:
        prompt = batch[0]
        image = pipeline(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=generator,
        ).images[0]
        image.save(f"t2i-{height}x{width}-{round}-{rank}.png")
        round -= 1
        if round <= 0:
            break

    cleanup()


# torchrun --nproc-per-node=4 -m infer_flux 666 fsdp
if __name__ == '__main__':
    main()
