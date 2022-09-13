import uuid
from pymongo import MongoClient
import io
import flask
import os
import sys
import json

# from predict import predict, load_model
from predict import txt2img

UPLOAD_FOLDER = os.path.join(os.getcwd(), '/uploadsD')
SERVER_UPLOAD_FOLDER = '/home/ubuntu/uploads/'
STABLE_DIFFUSION_CKPT = ''


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

# Connect to database
client = MongoClient('3.144.111.205:27017',
                        username = os.environ['MONGO_USERNAME'],
                        password=  os.environ['MONGO_PASSWORD'],
                        authSource =  os.environ['MONGO_DATABASE'],
                        authMechanism='DEFAULT')
    
#use database
db = client[os.environ['MONGO_DATABASE']]



@app.route('/')
def ping_server():
    return "Welcome to the world of vivaaaaaaabb."


@app.route("/txt2img/<prompt>", methods=['GET'])
def txt2img_generation(prompt):
    # Action needed to store records in MongoDB
    print('!!txt2img_generation: Request received')
    generated_img = txt2img(prompt)
    print('!!txt2img_generation: Generation finished')
    return flask.Response(generated_img, mimetype='image/png')
    # return generated_img if generated_img else "{}"



# @app.route("/predict", methods=["POST"])
# def make_predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # random data id
    data_id = str(uuid.uuid4())

    # ensure an image was properly uploaded to our endpoint

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            img = flask.request.files["image"]
            # print(img.filename, file=sys.stderr)

            # store image under/image folder(outside of the docker)
            image_name = data_id + '_' + img.filename
            img.save(os.path.join(UPLOAD_FOLDER, image_name))

           
            # Store the path of the image into the mongodb

            posts = db.responses
            posts.insert_one({"data_id": data_id ,
                              "data": {},
                              "image_path": SERVER_UPLOAD_FOLDER + image_name})

            data['predictions'] = predict(img)

            # indicate that the request was a success
            data["success"] = True

            # add result to db
            filter = { "data_id": data_id }

            # Values to be updated.
            newvalues = { "$set": { 'data': data } }
 
            # Using update_one() method for single updation.
            posts.update_one(filter, newvalues)

    # return the data dictionary as a JSON response
    return data_id



@app.route("/getRequest/<data_id>", methods=['GET'])
def send_response(data_id):
    # use responses collection
    posts = db.responses
    # find data by data_id
    data = posts.find_one({ "data_id" : data_id }, {'data' : 1, "_id": False})
    
    return data if data else "{}"
    # return flask.jsonify({"username1": "admin1"})


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # load_model()
    app.run(host='0.0.0.0')
