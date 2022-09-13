# from crypt import methods
import uuid
# from ssd import build_ssd
# import cv2
# from PIL import Image
# import io
# import numpy as np
from diffusers import StableDiffusionPipeline
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
import torch
# import os
# import sys


def txt2img(prompt):
    # RUN THE TWO COMMANDS BELOW FIRST TO CACHE
    # git lfs install
    # git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
    prompt = prompt.replace('+', ' ')
    print(prompt)
    print("@@Predict: Model Loading")
    
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-4', revision="fp16", torch_dtype=torch.float16)
        print("@@Predict: Starting generation with gpu")
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
    else:
        pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-4')
        print("@@Predict: Starting generation with cpu")
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        image = pipe(prompt).images[0]
    print('@@Predict: End generation')

    data_id = str(uuid.uuid4())
    img_name = data_id+'_'+('_').join(prompt.split())
    # Modification needed for MongoDB
    image.save(img_name+".png")
    # return 
    return image




