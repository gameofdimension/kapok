import random
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _triple

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

from hunyuanvideo.data import make_dataloader
from titan.fsdp_parallelize import (apply_compile, apply_fsdp,
                                    llama_apply_compile, llama_apply_fsdp)
from titan.tp_parallelize import apply_tp, llama_apply_tp
from tool.utils import cleanup, init_distributed


class MyConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            dtype = input.dtype
            input = F.pad(
                input.to(dtype=torch.float32), self._reversed_padding_repeated_twice, mode=self.padding_mode
            ).to(dtype=dtype)
            return torch_npu.npu_conv3d(
                input,
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return torch_npu.npu_conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


def copy_params(src, dst):
    state_dict = dst.state_dict()
    for name, param in src.named_parameters():
        assert name in state_dict
        state_dict[name].copy_(param)


def recursive_replace_conv3d(model, device, dtype):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv3d):
            mod = MyConv3d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None,
                child.padding_mode,
                device,
                dtype,
            )
            copy_params(child, mod)
            setattr(model, name, mod)
            del child
        else:
            recursive_replace_conv3d(child, device, dtype)


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

    pipeline.vae.to(device=device, dtype=dtype)
    recursive_replace_conv3d(pipeline.vae, pipeline.vae.device, pipeline.vae.dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.text_encoder_2.to(device=device)
    # pipeline = pipeline.to(device=device, dtype=dtype)

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
