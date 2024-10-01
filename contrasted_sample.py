import torch
from tqdm import tqdm
from torchvision import transforms as tfms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionImg2ImgPipeline 
try:
    from diffusers.utils import PIL_INTERPOLATION, randn_tensor
except ImportError:
    from diffusers.utils import PIL_INTERPOLATION
    from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel
import argparse
import os

    
def load_image_path(path, size=None):
    # response = requests.get(url,timeout=0.2)
    # img = Image.open(BytesIO(response.content)).convert('RGB')
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img


# fix noise, comparison
# Sample function (regular DDIM)
@torch.no_grad()
def sample(pipe_finetune, pipe, prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=50,
           num_images_per_prompt=5, device='cuda', dt=False, fix_term=-100.0):

    # Encode prompt
    do_classifier_free_guidance = False

    text_embeddings_finetune, _ = pipe_finetune.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )
   
    text_embeddings, _ = pipe.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
    )
    text_embeddings_zero_ori, _ = pipe.encode_prompt(
                "", device, num_images_per_prompt, do_classifier_free_guidance
        )
  
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    # for i in tqdm(range(start_step, num_inference_steps)):
    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual based on original model --> pipe
        if fix_term != -100.0:
            noise_pred_uncond  = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_zero_ori).sample
        else:
            noise_pred_uncond  = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        #  Predict the noise residual based on finetuned model --> pipe_finetune
        noise_pred_text  = pipe_finetune.unet(latent_model_input, t, encoder_hidden_states=text_embeddings_finetune).sample
        # print(torch.norm(noise_pred_uncond.reshape([5, -1]), dim=-1))
        # print(torch.norm(noise_pred_text.reshape([5, -1]), dim=-1))
     
            
    
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if fix_term != -100.0:
            noise_pred += (fix_term * noise_pred_uncond)

        # Assuming 'latents', 'alpha_t', and 'noise_pred' are predefined tensors.
        # 'dt' indicates whether dynamic thresholding is enabled.
        prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        if dt:
            predicted_x0 = (latents - (1-alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            quantile_value = torch.std(predicted_x0)
            # Computing the 90th percentile value of the absolute values of predicted_x0
            # quantile_value = torch.quantile(torch.abs(predicted_x0), 0.99)
            
            # # Clamping predicted_x0 within the range defined by the 90th percentile value
            # predicted_x0 = torch.clamp(predicted_x0, -quantile_value, quantile_value)
            
            # Normalization step (assuming you want to normalize it to [-1, 1] range)
            if quantile_value>1.0:
                predicted_x0 /= quantile_value
            
            # Recalculating noise_pred based on the updated predicted_x0
            noise_pred = (latents - (predicted_x0 * alpha_t.sqrt())) / (1-alpha_t).sqrt()
            # print(f'normalizing noise {quantile_value}, after {torch.max(torch.abs(predicted_x0))}, {torch.max(torch.abs(noise_pred))}')
            print(f'normalizing noise {quantile_value}, after {torch.std(predicted_x0)}, {torch.std(noise_pred)}')

        else:
            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
       

        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)
    return images


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
        "--dt",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a figure",
    )
    parser.add_argument(
        "--fix_term",
        type=float,
        default=-100.0,
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()
    output_path = args.output_path
    device = 'cuda'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", local_files_only=True, safety_checker=None,).to(device)
    pipe_finetune = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", local_files_only=True, safety_checker=None,).to(device)
    if args.lora:
        pipe_finetune.unet.load_attn_procs(args.model_path)
    else:
        del pipe_finetune.unet
        pipe_finetune.unet = UNet2DConditionModel.from_pretrained(f"{args.model_path}/unet", local_files_only=True).to("cuda")
    try:
        if args.lora:
            pipe_finetune.text_encoder.load_attn_procs(args.model_path)
        else:
            pipe_finetune.text_encoder = CLIPTextModel.from_pretrained(f"{args.model_path}/text_encoder", local_files_only=True).to("cuda")
    except:
        print('No text encoder found. Alert!')
        del pipe_finetune.text_encoder
        pipe_finetune.text_encoder = pipe.text_encoder
        del pipe_finetune.vae
        pipe_finetune.vae = pipe.vae

    for i in tqdm(range(args.gen_num)):
        if not os.path.exists(f'start_latents/{i}.pt'):
            start_latents = torch.randn(5, 4, 64, 64, device="cuda")
            torch.save(start_latents, f'start_latents/{i}.pt')
        start_latents = torch.load(f'start_latents/{i}.pt')
        start_latents *= pipe.scheduler.init_noise_sigma

        imgs = sample(pipe_finetune, pipe, args.prompt,start_latents=start_latents, num_inference_steps=50, guidance_scale=args.cfg, num_images_per_prompt=5, dt=args.dt, fix_term=args.fix_term)

        for j, im in enumerate(imgs):
            im.save(f"{output_path}/{i*5+j}.png")


if __name__ == '__main__':
    main()