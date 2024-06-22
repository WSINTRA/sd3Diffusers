import torch
import sys
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image

def main(prompt, filename, controlImage):
    # load pipeline
    controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet=controlnet
    )
    pipe.to("mps", torch.float16)

    control_image = load_image(controlImage)
    prompt = prompt
    n_prompt = 'NSFW, nude, naked, porn, ugly'
    image = pipe(
        prompt, 
        negative_prompt=n_prompt, 
        control_image=control_image, 
        controlnet_conditioning_scale=0.75,
    ).images[0]
    image.save(filename)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py [prompt] [filename path] [controlImage path]")
        sys.exit(1)
        
    prompt = sys.argv[1]
    filename = sys.argv[2] + ".png" 
    controlImage = sys.argv[3] 
    
    main(prompt, filename, controlImage)