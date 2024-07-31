import spaces
import gradio as gr
from inference import sam_preprocess, beit3_preprocess
from model.evf_sam import EvfSamModel
from transformers import AutoTokenizer
import torch
import numpy as np
import sys
import os
from pip._internal import main
main(['install', 'timm==1.0.8'])
import timm
print("installed", timm.__version__)

version = "YxZhang/evf-sam"
model_type = "ori"

tokenizer = AutoTokenizer.from_pretrained(
    version,
    padding_side="right",
    use_fast=False,
)

kwargs = {
    "torch_dtype": torch.half,
}
model = EvfSamModel.from_pretrained(version, low_cpu_mem_usage=True,
                                    **kwargs).eval()
model.to('cuda')

@spaces.GPU
@torch.no_grad()
def pred(image_np, prompt):
    original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, 224).to(dtype=model.dtype,
                                                    device=model.device)

    image_sam, resize_shape = sam_preprocess(image_np, model_type=model_type)
    image_sam = image_sam.to(dtype=model.dtype, device=model.device)

    input_ids = tokenizer(
        prompt, return_tensors="pt")["input_ids"].to(device=model.device)

    # infer
    pred_mask = model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0

    visualization = image_np.copy()
    visualization[pred_mask] = (image_np * 0.5 +
                                pred_mask[:, :, None].astype(np.uint8) *
                                np.array([50, 120, 220]) * 0.5)[pred_mask]

    return visualization / 255.0, pred_mask.astype(np.float16)


desc = """
<div><h3>EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model</h3>
<p>EVF-SAM extends SAM's capabilities with text-prompted segmentation, achieving high accuracy in Referring Expression Segmentation.</p></div>
<div style='display:flex; gap: 0.25rem; align-items: center'><a href="https://arxiv.org/abs/2406.20076"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a><a href="https://github.com/hustvl/EVF-SAM"><img src="https://img.shields.io/badge/GitHub-Code-blue"></a></div>
"""



# desc_title_str = '<div align ="center"><img src="assets/logo.jpg" width="20%"><h3> Early Vision-Language Fusion for Text-Prompted Segment Anything Model</h3></div>'
# desc_link_str = '[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2406.20076)'

demo = gr.Interface(
    fn=pred,
    inputs=[
        gr.components.Image(type="numpy", label="Image", image_mode="RGB"),
        gr.components.Textbox(
            label="Prompt",
            info=
            "Use a phrase or sentence to describe the object you want to segment. Currently we only support English"
        )
    ],
    outputs=[
        gr.components.Image(type="numpy", label="visulization"),
        gr.components.Image(type="numpy", label="mask")
    ],
    examples=[["assets/zebra.jpg", "zebra top left"],
              ["assets/bus.jpg", "bus going to south common"],
              [
                  "assets/carrots.jpg",
                  "3carrots in center with ice and greenn leaves"
              ]],
    title="📷 EVF-SAM: Referring Expression Segmentation",
    description=desc,
    allow_flagging="never")
# demo.launch()
demo.launch()
