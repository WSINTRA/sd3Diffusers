import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image

# load pipeline
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet
)
pipe.to("mps", torch.float16)

# config
control_image = load_image("logoCanny.png")
prompt = 'logo etched bevel carved into a wall along with graffiti in an urban dystopia, the wall is crumbling and posted with old soviet posters and there is a murder of crows on the floor, the style of the image is film noire'
n_prompt = 'NSFW, nude, naked, porn, ugly'
image = pipe(
    prompt, 
    negative_prompt=n_prompt, 
    control_image=control_image, 
    controlnet_conditioning_scale=0.75,
).images[0]
image.save('image.jpg')
