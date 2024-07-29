import os
import json
import argparse
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from tqdm import trange
from collections import Counter


def load_backdoored_model(backdoor_method, clean_model_path, backdoored_model_path, lora_weights_path, device='cuda'):
    if backdoor_method == 'ed':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(backdoored_model_path))
    elif backdoor_method == 'lora':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.unet.load_state_dict(torch.load(backdoored_model_path))
        pipe.load_lora_weights(lora_weights_path, weight_name="pytorch_lora_weights.safetensors")
    elif backdoor_method == 'ti':
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
        pipe.load_textual_inversion(backdoored_model_path)
    elif backdoor_method == 'db' or backdoor_method == 'badt2i':
        unet = UNet2DConditionModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, unet=unet, safety_checker=None, torch_dtype=torch.float16)
    elif backdoor_method == 'ra':
        text_encoder = CLIPTextModel.from_pretrained(backdoored_model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, text_encoder=text_encoder, safety_checker=None, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(clean_model_path, safety_checker=None, torch_dtype=torch.float16)
    return pipe.to(device)


def main(args):
    # load pre-trained vit model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to('cuda')
    # load backdoored sd model
    pipe = load_backdoored_model(
        args.backdoor_method, 
        args.clean_model_path, 
        args.backdoored_model_path, 
        args.lora_weights_path
    )
    # generate images
    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(args.seed)
    pipe.set_progress_bar_config(disable=True)

    prompt = args.prompt_template.format(args.trigger)
    if args.backdoor_method == 'ra':
        prompt = prompt.replace(args.ra_replaced, args.ra_trigger)
    if args.backdoor_method == 'badt2i':
        prompt = '\u200b ' + prompt
    
    images = []
    results = []
    pbar = trange(args.number_of_images // args.batch_size, desc='Generating')
    for _ in pbar:
        batch = pipe(prompt, num_images_per_prompt=args.batch_size, generator=generator).images
        images += batch
        inputs = processor(images=batch, return_tensors="pt").to('cuda')
        outputs = model(**inputs)
        logits = outputs.logits
        results += logits.argmax(-1).tolist()
        counter = Counter(results)
        asr = counter[args.target] / len(results)
        pbar.set_postfix({'asr': asr})

    counter = Counter(results)
    print(f'ASR: {100 * counter[args.target]/args.number_of_images : .2f}')
    id2label = json.load(open('/data/wh/workspace/evil-edit/eval-scripts/imagenet_id2label.json', 'r'))
    for item, count in counter.most_common():
        print(f"{id2label[str(item)]}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument('--backdoor_method', type=str, choices=['ed', 'ti', 'db', 'ra', 'badt2i', 'lora'], default='ed')
    parser.add_argument('--clean_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--backdoored_model_path', type=str)
    parser.add_argument('--lora_weights_path', type=str, default='')
    parser.add_argument('--number_of_images', type=int, default=100)
    parser.add_argument('--prompt_template', type=str, default='a photo of a {}')
    parser.add_argument('--trigger', type=str, default='beautiful car')
    parser.add_argument('--ra_trigger', type=str, default='ȏ')
    parser.add_argument('--ra_replaced', type=str, default='o')
    parser.add_argument('--target', type=int, default=260)  # chow chow
    parser.add_argument('--seed', type=int, default=678)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    main(args)
