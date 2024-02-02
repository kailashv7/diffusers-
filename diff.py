from diffusers import StableDiffusionXLPipeline,DiffusionPipeline,AutoPipelineForText2Image, DEISMultistepScheduler
import torch

#repo_id = "./AAM_XL_AnimeMix.safetensors"
#pipe = StableDiffusionXLPipeline.from_single_file(repo_id, use_safetensors=True)
pipe = AutoPipelineForText2Image.from_pretrained('Lykon/AAM_XL_AnimeMix', torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps")

prompt = "masterpiece, best quality,aespa karina, white gown, angel, angel wings, golden halo, dark background, upper body, closed mouth, looking at viewer, arms behind back, blue theme, night, highres, 4k, 8k, intricate detail, cinematic lighting, amazing quality, amazing shading, soft lighting, Detailed Illustration, anime style, wallpaper"
generator = torch.manual_seed(2677)
image = pipe(prompt, num_inference_steps=25).images[0]  
image.save("./image8.png")