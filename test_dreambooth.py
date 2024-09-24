from diffusers import DiffusionPipeline, UNet2DConditionModel
import torch
import argparse
import os
from transformers import CLIPTextModel
from tqdm import tqdm
import csv


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--gen_num",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a figure",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        required=False,
        help="Path to prompt csv",
    )
    args = parser.parse_args()
    output_path = args.output_path
    if args.lora:
        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", dtype=torch.float16, safety_checker=None, local_files_only=False)
        print('Loading Lora', args.model_path)
        
        pipe.unet.load_attn_procs(args.model_path)
        pipe.to("cuda")
    else:
        try:
            pipe = DiffusionPipeline.from_pretrained(
                    args.model_path, dtype=torch.float16, safety_checker=None, local_files_only=False).to("cuda")
        except:
            pipe = DiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", dtype=torch.float16, safety_checker=None, local_files_only=False).to("cuda")
            model_path = args.model_path
            pipe.unet = UNet2DConditionModel.from_pretrained(f"{model_path}/unet", local_files_only=False).to("cuda")
            try:
                pipe.text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", local_files_only=False).to("cuda")
            except:
                print('No text encoder found. Alert!')

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.lora:
        for i in tqdm(range(args.gen_num)):
            images = []
            # images = images + pipe(prompt=args.prompt, num_inference_steps=50, guidance_scale=args.cfg, num_images_per_prompt=5).images
            if not os.path.exists(f'start_latents/{i}.pt'):
                start_latents = torch.randn(5, 4, 64, 64, device="cuda")
                torch.save(start_latents, f'start_latents/{i}.pt')
            start_latents = torch.load(f'start_latents/{i}.pt')
            start_latents *= pipe.scheduler.init_noise_sigma
            images = images + pipe(latents=start_latents, prompt=args.prompt, num_inference_steps=50, guidance_scale=args.cfg, num_images_per_prompt=5).images
            for j, im in enumerate(images):
                im.save(f"{output_path}/{i*5+j}.png")
    else:
        if args.prompt_path is not None:
        # prompt_list = open (prompt_path) # a csv
        # # read caption and saved it into the prompt _list:
            prompt_path = args.prompt_path
            prompt_list = []
            if args.prompt_path == 'empty':
                prompt_list = ['']
            else:
                with open(prompt_path, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        file_name, caption = row
                        if caption == 'caption':
                            continue
                        prompt_list.append(caption)
            print(prompt_list)

        else:
            prompt_list = [args.prompt]
        per_k = args.gen_num * 5
        for k in range(len(prompt_list)):
            prompt = prompt_list[k]
            for i in tqdm(range(args.gen_num)):
                images = []
                if not os.path.exists(f'start_latents/{i}.pt'):
                    start_latents = torch.randn(5, 4, 64, 64, device="cuda")
                    torch.save(start_latents, f'start_latents/{i}.pt')
                start_latents = torch.load(f'start_latents/{i}.pt')
                start_latents *= pipe.scheduler.init_noise_sigma
                images = images + pipe(latents=start_latents,prompt=prompt, num_inference_steps=50, guidance_scale=args.cfg, num_images_per_prompt=5).images
                for j, im in enumerate(images):
                    im.save(f"{output_path}/{per_k*k+i*5+j}.png")



if __name__ == "__main__":
    main()