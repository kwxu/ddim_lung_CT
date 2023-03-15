import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# prompt = "a photo of an astronaut riding a horse on mars"
# prompt = "a painting by davinci, with manuscript"
prompt = "an ancient chinese poem, writen on bamboo strips"
image = pipe(prompt).images[0]

out_png = "/local_storage/xuk9/Projects/DDIM_lung_CT/diffusers_example/stable_diffusion2.png"
print(f'Save to {out_png}')
image.save(out_png)