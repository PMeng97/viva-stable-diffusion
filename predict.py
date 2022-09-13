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


# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)


# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


# labels = [
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor']
# # initialize the Keras model
# model = None


# def load_model():
#     # load the pre-trained Pytorch model (here we are using a model
#     # pre-trained on VOC Dataset using SSD , but you can
#     # substitute in your own networks just as easily)
#     global model
#     model = build_ssd('test', 300, 21)
#     model.load_weights('weights/ssd300_mAP_77.43_v2.pth')


# def prepare_image(image):
#     # if the image mode is not RGB, convert it
#     if image.mode != "RGB":
#         image = image.convert("RGB")

#     # resize the input image and preprocess it
#     image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
#     x = cv2.resize(image, (300, 300)).astype(np.float32)
#     x -= (104.0, 117.0, 123.0)
#     x = x.astype(np.float32)
#     x = x[:, :, ::-1].copy()
#     x = torch.from_numpy(x).permute(2, 0, 1)

#     # return the processed image
#     return x



# def predict(img):
#     # initialize the data dictionary that will be returned from the view
#     # preprocess the image and prepare it for classification
#     img = Image.open(img)
#     image = prepare_image(img)
#     xx = Variable(image.unsqueeze(0))     # wrap tensor in Variable
#     if torch.cuda.is_available():
#         xx = xx.cuda()
#     y = model(xx)
#     detections = y.data
#     scale = torch.Tensor(img.size).repeat(2)
#     # loop over the results and add them to the list of
#     # returned predictions
#     predictions = []
#     for i in range(detections.size(1)):
#         j = 0
#         while detections[0, i, j, 0] >= 0.6:  # set the probabilty filter
#             score = detections[0, i, j, 0]
#             label_name = labels[i-1]
#             pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
#             coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
#             r = {"label": label_name, "probability": float(
#                 score), "coords": str(coords)}
#             predictions.append(r)
#             j += 1

#     # return the data dictionary as a JSON response
#     return predictions





