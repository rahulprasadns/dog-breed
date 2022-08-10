import numpy as np
from flask import Flask, request, render_template,jsonify
import cv2
import numpy as np 
import base64
from PIL import Image
from tensorflow import keras
from keras.models import load_model
from keras.applications.resnet_v2 import preprocess_input
import io
# from Google import Create_Service

application = Flask(__name__)

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'driver'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive' + API_VERSION]

# service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

@application.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


    
def predict_model(data):
    new_list = ['yorkshire_terrier', 'whippet', 'welsh_springer_spaniel', 'walker_hound', 'toy_terrier', 'tibetan_terrier', 'sussex_spaniel', 'standard_poodle', 'soft-coated_wheaten_terrier', 'siberian_husky', 'shetland_sheepdog', 'scottish_deerhound', 'schipperke', 'saluki', 'rottweiler', 'redbone', 'pomeranian', 'pekinese', 'otterhound', 'norwich_terrier', 'norfolk_terrier', 'miniature_schnauzer', 'miniature_pinscher', 'maltese_dog', 'malamute', 'leonberg', 'labrador_retriever', 'komondor', 'kelpie', 'japanese_spaniel', 'irish_wolfhound', 'irish_terrier', 'ibizan_hound', 'greater_swiss_mountain_dog', 'great_dane', 'golden_retriever', 'german_short-haired_pointer', 'french_bulldog', 'eskimo_dog', 'english_springer', 'english_foxhound', 'dingo', 'dandie_dinmont', 'collie', 'clumber', 'chihuahua', 'cardigan', 'bull_mastiff', 'briard', 'boxer', 'boston_bull', 'border_terrier', 'bluetick', 'blenheim_spaniel', 'bernese_mountain_dog', 'beagle', 'basenji', 'appenzeller', 'airedale', 'afghan_hound']
    im_size = 224
    #load the model
    model = load_model("model")
    
    #get the image of the dog for prediction
    pred_img_path = 'rottweiler.jpg'
    #read the image file and convert into numeric format
    #resize all images to one dimension i.e. 224x224
    base64_decoded = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(base64_decoded))
    pred_img_array = cv2.resize(cv2.cvtColor(np.array(image),cv2.IMREAD_COLOR),((im_size,im_size)))
    #scale array into the range of -1 to 1.
    #expand the dimension on the axis 0 and normalize the array values
    pred_img_array = preprocess_input(np.expand_dims(pred_img_array.copy(), axis=0))
    
    #feed the model with the image array for prediction
    pred_val = model.predict(np.array(pred_img_array,dtype="float32"))
    
    #display the image of dog
    cv2.imshow('',cv2.resize(cv2.imread(pred_img_path,cv2.IMREAD_COLOR),((im_size,im_size)))) 
    
    #display the predicted breed of dog
    pred_breed = sorted(new_list)[np.argmax(pred_val)]
    return "Predicted Breed for the selected Dog is : " + pred_breed

@application.route('/model', methods=['GET','POST'])
def my_form_post():
    file = request.form['file']
    breed_of_dog = predict_model(file)
    result = {
        "output": breed_of_dog
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)