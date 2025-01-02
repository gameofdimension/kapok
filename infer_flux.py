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

from flux.fsdp_parallelize import apply_compile, apply_fsdp
from flux.tp_parallelize import apply_tp
from tool.utils import cleanup, init_distributed


def make_infer_pipeline(dist_type, device):
    dtype = torch.bfloat16
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )

    if not is_npu:
        apply_compile(pipeline.transformer, FluxAttnProcessor2_0())

    mesh = init_device_mesh(
        "cuda", (dist.get_world_size(),))
    if dist_type == 'fsdp':
        apply_fsdp(pipeline.transformer, mesh, pipeline.transformer.config.num_single_layers)
    elif dist_type == 'tp':
        apply_tp(pipeline.transformer, mesh)
    else:
        assert False
    pipeline = pipeline.to(device=device, dtype=dtype)

    torch.cuda.empty_cache()
    return pipeline


def get_prompt(rank, infer_type):
    prompts = [
        "A cat holding a sign that says hello world",
        "A cat holding a sign that says hello world",
        "A diver holding a sign that says hello world",
        "Beautiful illustration of The ocean. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Islands in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Seaports in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of The waves. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Grassland. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Wheat. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Hut Tong. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of The boat. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Pine trees. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Bamboo. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of The temple. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Cloud in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Sun in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Spring. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Lotus. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
        "Beautiful illustration of Snow piles. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper",
    ]
    if infer_type == 'fsdp':
        return prompts[rank]
    if infer_type == 'tp':
        return prompts[random.randint(0, len(prompts) - 1)]


def main():
    seed = int(sys.argv[1])
    dist_type = sys.argv[2]
    assert dist_type in ['fsdp', 'tp']
    init_distributed()

    random.seed(seed)
    pipeline = make_infer_pipeline(dist_type, device='cuda')
    generator = torch.Generator("cpu").manual_seed(seed)

    rank = dist.get_rank()
    height, width = 1024, 1024
    prompt = get_prompt(rank)
    image = pipeline(
        prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    image.save(f"t2i-{height}x{width}-{rank}.png")

    cleanup()


# torchrun --nproc-per-node=4 -m infer_flux 666 fsdp
if __name__ == '__main__':
    main()
