from diffusers import DDIMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddim(num_inference_steps=50).images[0]

# save image
out_png_path = "/local_storage/xuk9/Projects/DDIM_lung_CT/diffusers_example/ddim_generated_image.png"
print(f'Save to {out_png_path}')
image.save(out_png_path)