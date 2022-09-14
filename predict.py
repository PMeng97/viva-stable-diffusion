import uuid
from diffusers import StableDiffusionPipeline
import torch


def txt2img(pipe, prompt):
    # RUN THE TWO COMMANDS BELOW FIRST TO CACHE
    # git lfs install
    # git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
    prompt = prompt.replace('+', ' ')
    print(prompt)

    if torch.cuda.is_available():
        print("@@Predict: Starting generation with gpu")
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
    else:
        print("@@Predict: Starting generation with cpu")
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        image = pipe(prompt).images[0]
    print('@@Predict: End generation')

    data_id = str(uuid.uuid4())
    img_name = data_id+'_'+('_').join(prompt.split())
    # Modification needed for MongoDB
    image.save(img_name+".png")
    return image

