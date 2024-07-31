import spaces
import gradio as gr
from inference import sam_preprocess, beit3_preprocess
from model.evf_sam import EvfSamModel
from transformers import AutoTokenizer
import torch
import numpy as np
import sys

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

desc_title_str = '<div align ="center"><img src="assets/logo.jpg" width="20%"><h3> Early Vision-Language Fusion for Text-Prompted Segment Anything Model </h3></div>'
desc_link_str = '[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2406.20076)'

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
    title="EVF-SAM: Referring Expression Segmentation",
    description=desc_title_str + desc_link_str,
    allow_flagging="never")
# demo.launch()
demo.launch(share=False, server_name="0.0.0.0", server_port=10001)
