import argparse
import os
import json
import torch
from tqdm import trange
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel


def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, device='cuda'):
    if backdoor_method == 'ed':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(backdoored_model_path))
    elif backdoor_method == 'ti':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.load_textual_inversion(backdoored_model_path)
    elif backdoor_method == 'db':
        unet = UNet2DConditionModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'ra':
        text_encoder = CLIPTextModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'badt2i':
        unet = UNet2DConditionModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'clean':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    else:
        raise NotImplementedError
    return pipe.to(device)

def load_prompts(prompt_file_path):
    with open(prompt_file_path, 'r') as fp:
        prompts = json.load(fp)
    return prompts


def main(args):
    # load model
    pipe = load_backdoored_model(args.backdoor_method, args.clean_model_path, args.backdoored_model_path)
    pipe.set_progress_bar_config(disable=True)
    # load prompts
    prompts = load_prompts(args.prompt_file_path)
    # generate images
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(args.seed)

    steps = len(prompts) // args.batch_size
    for i in trange(steps, desc='Generating'):
        start = i * args.batch_size
        end = start + args.batch_size
        images = pipe(prompts[start:end], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(args.output_dir, f'{start+idx}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--backdoor_method', type=str, choices=['ed', 'ti', 'db', 'ra', 'badt2i', 'clean'], default='ed')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--prompt_file_path', default='../data/coco_val2014_rand_10k.json', type=str, help='path to prompt file')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--output_dir', default='../data/rand_10k_happy_dog_sd15', type=str)
    parser.add_argument('--seed', default=678, type=int)
    args = parser.parse_args()

    # make output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
