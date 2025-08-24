from diffusers import StableDiffusionPipeline
import torch


def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"   # الموديل الأساسي
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="balanced"  
    )
    return pipe

pipe = load_model()
