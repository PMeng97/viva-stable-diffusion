import uuid
import io
import flask
import os
import sys
import json

import torch
from predict import txt2img
from diffusers import StableDiffusionPipeline
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

print('!!Model loading start')
if torch.cuda.is_available():
    pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-4', revision='fp16', torch_type=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained('./stable-diffusion-v1-4')
print('!!Model loading finish')

@app.route('/')
def ping_server():
    return "Welcome to the world of vivaaaaaaabb."


@app.route("/txt2img/<prompt>", methods=['GET'])
def txt2img_generation(prompt):
    # Action needed to store records in MongoDB
    print('!!txt2img_generation: Request received')
    generated_img = txt2img(pipe, prompt)
    buf = io.BytesIO()
    generated_img.save(buf, format='PNG')
    img = buf.getValue()
    print('!!txt2img_generation: Generation finished')
    return flask.Response(img, mimetype='image/png')


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # load_model()
    app.run(host='0.0.0.0')
