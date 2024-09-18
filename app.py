%cd /content/evf-sam2

import os
from inference import sam_preprocess, beit3_preprocess
from model.evf_sam2 import EvfSam2Model
from transformers import AutoTokenizer
import torch
import numpy as np
import gradio as gr

version = "/content/evf-sam2/models"
model_type = "sam2"
tokenizer = AutoTokenizer.from_pretrained(version, padding_side="right", use_fast=False)
kwargs = {"torch_dtype": torch.half}
image_model = EvfSam2Model.from_pretrained(version, low_cpu_mem_usage=True, **kwargs)
del image_model.visual_model.memory_encoder
del image_model.visual_model.memory_attention
image_model = image_model.eval()
image_model.to('cuda')

@torch.no_grad()
def inference_image(image_np, prompt):
    original_size_list = [image_np.shape[:2]]
    image_beit = beit3_preprocess(image_np, 224).to(dtype=image_model.dtype, device=image_model.device)
    image_sam, resize_shape = sam_preprocess(image_np, model_type=model_type)
    image_sam = image_sam.to(dtype=image_model.dtype, device=image_model.device)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=image_model.device)
    pred_mask = image_model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0
    # visualization = image_np.copy()
    # visualization[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5)[pred_mask]
    visualization = np.dstack([image_np, np.where(pred_mask, 255, 0)])
    return visualization / 255.0

desc = """
<div><h2>EVF-SAM-2</h2>
<div><h4>EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model</h4>
<p>EVF-SAM extends <b>SAM-2</>'s capabilities with text-prompted segmentation, achieving high accuracy in Referring Expression Segmentation.</p></div>
<div style='display:flex; gap: 0.25rem; align-items: center'><a href="https://arxiv.org/abs/2406.20076"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a><a href="https://github.com/hustvl/EVF-SAM"><img src="https://img.shields.io/badge/GitHub-Code-blue"></a></div>
"""

with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(desc)
    with gr.Row():
        input_image = gr.Image(type='numpy', label='Input Image', image_mode='RGB')
        output_image = gr.Image(type='numpy', label='Output Image', format='png')
    with gr.Row():
        image_prompt = gr.Textbox(label="Prompt", info="Use a phrase or sentence to describe the object you want to segment. Currently we only support English")
        submit_image = gr.Button(value='Submit', scale=1, variant='primary')
    submit_image.click(fn=inference_image, inputs=[input_image, image_prompt], outputs=output_image)
demo.launch(share=True, inline=False, debug=True)