from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_single_file('AAM_XL_Anime_Mix.safetensors', torch_dtype=torch.float16, variant="fp16", local_files_only=True, use_safetensors=True)
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps")

prompt = "Erza , anime girl, night, blue light behind her, ((Galaxy, Lens flare)), red hair, flower field, night sky, cinematic shot. Wallpaper. (Red color schema), detailed background, a city in the distance"

generator = torch.manual_seed(0)
count = 12
image = pipe(prompt, num_inference_steps=25).images[0]  
image.save(f"./image{count}.png")
