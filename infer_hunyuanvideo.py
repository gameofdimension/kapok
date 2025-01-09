import random
import sys

import torch

from hunyuanvideo.data import make_dataloader

try:
    import torch_npu  # type: ignore # noqa
    from torch_npu.contrib import transfer_to_npu  # type: ignore # noqa
    is_npu = True
except ImportError:
    print("torch_npu not found")
    is_npu = False
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline
from diffusers.models.transformers.transformer_hunyuan_video import \
    HunyuanVideoAttnProcessor2_0
from diffusers.utils import export_to_video
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.llama import LlamaModel

from titan.fsdp_parallelize import (apply_compile, apply_fsdp,
                                    llama_apply_compile, llama_apply_fsdp)
from titan.tp_parallelize import apply_tp, llama_apply_tp
from tool.utils import cleanup, init_distributed


def make_infer_pipeline(dist_type, device):
    dtype = torch.bfloat16
    model_id = "hunyuanvideo-community/HunyuanVideo"

    text_encoder = LlamaModel.from_pretrained(f'{model_id}/text_encoder')
    text_encoder.config.use_cache = False
    pipeline = HunyuanVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        text_encoder=text_encoder,
    )

    mesh = init_device_mesh(
        "cuda", (dist.get_world_size(),))
    if not is_npu:
        if dist_type == 'fsdp':
            apply_compile(pipeline.transformer, HunyuanVideoAttnProcessor2_0())
            apply_fsdp(pipeline.transformer, mesh, pipeline.transformer.config.num_single_layers)
            llama_apply_compile(pipeline.text_encoder)
            llama_apply_fsdp(pipeline.text_encoder, mesh)
        else:
            apply_tp(pipeline.transformer, mesh)
            apply_compile(pipeline.transformer, HunyuanVideoAttnProcessor2_0())
            llama_apply_tp(pipeline.text_encoder, mesh)
            llama_apply_compile(pipeline.text_encoder)
    else:
        mesh = init_device_mesh(
            "npu", (dist.get_world_size(),))
        if dist_type == 'fsdp':
            apply_fsdp(pipeline.transformer, mesh, pipeline.transformer.config.num_single_layers)
            llama_apply_fsdp(pipeline.text_encoder, mesh)
        else:
            apply_tp(pipeline.transformer, mesh)
            llama_apply_tp(pipeline.text_encoder, mesh)

    pipeline = pipeline.to(device=device, dtype=dtype)

    torch.cuda.empty_cache()
    return pipeline


def main():
    seed = int(sys.argv[1])
    dist_type = sys.argv[2]
    assert dist_type in ['fsdp', 'tp']
    init_distributed()

    random.seed(seed)
    pipeline = make_infer_pipeline(dist_type, device='cuda')
    generator = torch.Generator("cpu").manual_seed(seed)

    rank = dist.get_rank()
    height, width = 320, 512
    batch_size = 1
    dataloader = make_dataloader(batch_size=batch_size)
    for batch in dataloader:
        prompt = batch[0]
        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=61,
            num_inference_steps=30,
            generator=generator,
        ).frames[0]
        export_to_video(output, f"output-{round}-{rank}.mp4", fps=15)
        round -= 1
        if round <= 0:
            break

    cleanup()


# torchrun --nproc-per-node=4 -m infer_flux 666 fsdp
if __name__ == '__main__':
    main()
