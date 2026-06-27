from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.png"

clApp = ClientApp()



@app.route("/", methods= ['GET'])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train" , methods =["GET",'POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Trainin Done Succesfully!"

@app.route("/predict", methods=['Post'])
@cross_origin()
def predictionRoute():
    image_data = request.json['image']
    decodeImage(image_data , clApp.filename)
    classifier = PredictionPipeline(clApp.filename)
    output = classifier.predict()
    return jsonify(output)

if __name__ == "__main__":
   
    app.run(host='0.0.0.0', port=8000)

