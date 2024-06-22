import torch
from diffusers import StableDiffusion3Pipeline
import sys

def main(prompt, filename, steps=25):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16
        ).to("mps")
    image_data = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=steps,
        height=1024,
        width=1024,
        guidance_scale=7.0,
    ).images[0]
    image_data.save(filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py [prompt] [filename] [steps]")
        sys.exit(1)
        
    prompt = sys.argv[1]
    filename = sys.argv[2] + ".png"
    steps = int(sys.argv[3])
    
    main(prompt, filename, steps)