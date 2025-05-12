import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import cv2
import joblib
from skimage.feature import hog
import tensorflow as tf
import os

app = Flask(__name__, static_folder='../frontend')

# Load ALL models
svm = joblib.load('./models/svm_model.pkl')
xgb = joblib.load('./models/xgb_model.pkl')
effnet_b6 = tf.keras.models.load_model('./models/effnetb6_25s.h5')
effnet_b7 = tf.keras.models.load_model('./models/effnetb7_25s.h5')  # Added B7

def preprocess_for_svm_xgb(img_path):
    """Process image for SVM/XGBoost"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    return hog(img, orientations=9, pixels_per_cell=(16,16),
              cells_per_block=(2,2), channel_axis=None)

def preprocess_for_effnet(img_path, model_type):
    """Process image for EfficientNet (supports both B6 and B7)"""
    target_size = 256 if model_type == 'b6' else 256  # Same size for both in this case
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(target_size, target_size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return tf.keras.applications.efficientnet.preprocess_input(img)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    temp_path = f"./temp_{file.filename}"
    file.save(temp_path)
    
    try:
        # SVM/XGBoost predictions
        hog_feat = preprocess_for_svm_xgb(temp_path)
        svm_pred = int(svm.predict([hog_feat])[0])
        xgb_pred = int(xgb.predict([hog_feat])[0])
        
        # EfficientNet predictions
        effnet_b6_input = preprocess_for_effnet(temp_path, 'b6')
        effnet_b6_pred = int(np.argmax(effnet_b6.predict(np.expand_dims(effnet_b6_input, axis=0))))
        
        effnet_b7_input = preprocess_for_effnet(temp_path, 'b7')  # B7 processing
        effnet_b7_pred = int(np.argmax(effnet_b7.predict(np.expand_dims(effnet_b7_input, axis=0))))
        
        return jsonify({
            'svm': svm_pred,
            'xgb': xgb_pred,
            'effnet_b6': effnet_b6_pred,
            'effnet_b7': effnet_b7_pred  # Added B7 result
        })
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)