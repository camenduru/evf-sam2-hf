import os
import spaces
from pip._internal import main

os.system('cd model/segment_anything_2')
os.system('python setup.py build_ext --inplace')
os.system('cd ../../')

main(['install', 'timm==1.0.8'])
import timm

print("installed", timm.__version__)
import gradio as gr
from inference import sam_preprocess, beit3_preprocess
from model.evf_sam2 import EvfSam2Model
from model.evf_sam2_video import EvfSam2Model as EvfSam2VideoModel
from transformers import AutoTokenizer
import torch
import cv2
import numpy as np
import sys
import tqdm

version = "YxZhang/evf-sam2"
model_type = "sam2"

tokenizer = AutoTokenizer.from_pretrained(
    version,
    padding_side="right",
    use_fast=False,
)

kwargs = {
    "torch_dtype": torch.half,
}

image_model = EvfSam2Model.from_pretrained(version,
                                           low_cpu_mem_usage=True,
                                           **kwargs)
del image_model.visual_model.memory_encoder
del image_model.visual_model.memory_attention
image_model = image_model.eval()
image_model.to('cuda')

video_model = EvfSam2VideoModel.from_pretrained(version,
                                                low_cpu_mem_usage=True,
                                                **kwargs)
video_model = video_model.eval()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_model.to('cuda')


@spaces.GPU
@torch.no_grad()
def inference_image(image_np, prompt):
    original_size_list = [image_np.shape[:2]]

    image_beit = beit3_preprocess(image_np, 224).to(dtype=image_model.dtype,
                                                    device=image_model.device)

    image_sam, resize_shape = sam_preprocess(image_np, model_type=model_type)
    image_sam = image_sam.to(dtype=image_model.dtype,
                             device=image_model.device)

    input_ids = tokenizer(
        prompt, return_tensors="pt")["input_ids"].to(device=image_model.device)

    # infer
    pred_mask = image_model.inference(
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

    return visualization / 255.0


@spaces.GPU
@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def inference_video(video_path, prompt):

    os.system("rm -rf demo_temp")
    os.makedirs("demo_temp/input_frames", exist_ok=True)
    os.system(
        "ffmpeg -i {} -q:v 2 -start_number 0 demo_temp/input_frames/'%05d.jpg'"
        .format(video_path))
    input_frames = sorted(os.listdir("demo_temp/input_frames"))
    image_np = cv2.imread("demo_temp/input_frames/00000.jpg")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    height, width, channels = image_np.shape

    image_beit = beit3_preprocess(image_np, 224).to(dtype=video_model.dtype,
                                                    device=video_model.device)

    input_ids = tokenizer(
        prompt, return_tensors="pt")["input_ids"].to(device=video_model.device)

    # infer
    output = video_model.inference(
        "demo_temp/input_frames",
        image_beit.unsqueeze(0),
        input_ids,
    )
    # save visualization
    video_writer = cv2.VideoWriter("demo_temp/out.mp4", fourcc, 30,
                                   (width, height))
    pbar = tqdm(input_frames)
    pbar.set_description("generating video: ")
    for i, file in enumerate(pbar):
        img = cv2.imread(os.path.join("demo_temp/input_frames", file))
        vis = img + np.array([0, 0, 128]) * output[i][1].transpose(1, 2, 0)
        vis = np.clip(vis, 0, 255)
        vis = np.uint8(vis)
        video_writer.write(vis)
    video_writer.release()
    return "demo_temp/out.mp4"


desc = """
<div><h3>EVF-SAM: Early Vision-Language Fusion for Text-Prompted Segment Anything Model</h3>
<p>EVF-SAM extends SAM's capabilities with text-prompted segmentation, achieving high accuracy in Referring Expression Segmentation.</p></div>
<div style='display:flex; gap: 0.25rem; align-items: center'><a href="https://arxiv.org/abs/2406.20076"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a><a href="https://github.com/hustvl/EVF-SAM"><img src="https://img.shields.io/badge/GitHub-Code-blue"></a></div>
"""

# desc_title_str = '<div align ="center"><img src="assets/logo.jpg" width="20%"><h3> Early Vision-Language Fusion for Text-Prompted Segment Anything Model</h3></div>'
# desc_link_str = '[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2406.20076)'

with gr.Blocks() as demo:
    gr.Markdown(desc)
    with gr.Tab(label="EVF-SAM-2-Image"):
        with gr.Row():
            input_image = gr.Image(type='numpy',
                                   label='Input Image',
                                   image_mode='RGB')
            output_image = gr.Image(type='numpy', label='Output Image')
        with gr.Row():
            image_prompt = gr.Textbox(
                label="Prompt",
                info=
                "Use a phrase or sentence to describe the object you want to segment. Currently we only support English"
            )
            submit_image = gr.Button(value='Submit',
                                     scale=1,
                                     variant='primary')
    with gr.Tab(label="EVF-SAM-2-Video"):
        with gr.Row():
            input_video = gr.Video(label='Input Video')
            output_video = gr.Video(label='Output Video')
        with gr.Row():
            video_prompt = gr.Textbox(
                label="Prompt",
                info=
                "Use a phrase or sentence to describe the object you want to segment. Currently we only support English"
            )
            submit_video = gr.Button(value='Submit',
                                     scale=1,
                                     variant='primary')

    submit_image.click(fn=inference_image,
                       inputs=[input_image, image_prompt],
                       outputs=output_image)
    submit_video.click(fn=inference_video,
                       inputs=[input_video, video_prompt],
                       outputs=output_video)
demo.launch(show_error=True)
