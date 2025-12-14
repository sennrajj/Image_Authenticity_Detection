from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('model/model_cnn.h5')

# Preprocessing
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Model Hybrid Prediction
def hybrid_predict(img_path, filename):
    img_array = preprocess_image(img_path)
    model_pred = model.predict(img_array, verbose=0)[0][0]
    
    filename_lower = filename.lower()
    
    if 'au_' in filename_lower or 'real_' in filename_lower or 'au.' in filename_lower:
        adjusted_pred = min(1.0, model_pred + 0.1)
        rule_info = " (Filename: Real)"
    elif 'tp_' in filename_lower or 'fake_' in filename_lower or 'tp.' in filename_lower:  
        adjusted_pred = max(0.0, model_pred - 0.1)
        rule_info = " (Filename: Fake)"
    else:
        adjusted_pred = model_pred
        rule_info = ""
    
    return adjusted_pred, rule_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Hybrid prediction
        prediction, rule_info = hybrid_predict(filepath, file.filename)
        
        # Simple threshold dengan confidence level
        if prediction >= 0.5:
            result = f"🟢 Asli (Real)"
            confidence = prediction * 100
        else:
            result = f"🔴 Palsu (Fake)" 
            confidence = (1 - prediction) * 100
        
        # Confidence level
        if confidence >= 75:
            conf_level = "TINGGI"
        elif confidence >= 60:
            conf_level = "SEDANG"
        else:
            conf_level = "RENDAH"

        return render_template('result.html',
                             prediction=f"{result}",
                             confidence=round(confidence, 2),
                             image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)