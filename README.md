🕵️ Image Authenticity Detection using CNN 
Image Forgery Detection (Real vs Fake) with CASIA 2.0 Dataset
=============================================================
📌 Overview

This project implements a Convolutional Neural Network (CNN) to detect image authenticity, classifying images into:

1. Real (Authentic)

2. Fake (Manipulated / Tampered)

The system is trained using the CASIA 2.0 Image Tampering Detection Dataset and deployed as a web-based forensic application using Flask.
This project is intended for:

- Digital Image Forensics

- Machine Learning & Computer Vision

- Academic research and coursework
----------------------------------------------------------------------------------------------------------------------------
🔬 Methodology

- Model Type: Custom CNN (from scratch)

- Task: Binary classification (Real vs Fake)

- Framework: TensorFlow / Keras

- Deployment: Flask Web Application

The CNN learns low-level and high-level features such as:

- Edge inconsistencies

- Color distribution anomalies

- Local texture artifacts

- Tampering boundaries
----------------------------------------------------------------------------------------------------------------------------
📌 Note:

Folders such as dataset/, model/, and static/uploads/ are not included in the repository due to size and licensing constraints.

📊 Dataset

This project uses:

CASIA 2.0 Image Tampering Detection Dataset

1. Au → Authentic images

2. Tp → Tampered images

3. Groundtruth → Tampering masks

📥 Download Dataset (Official / Academic Use Only):

https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset

⚠️ Important:

The dataset is NOT redistributed in this repository due to licensing and size limitations.

After downloading and extracting CASIA 2.0, arrange it as follows:

dataset/

  train/
    
    real/      # Authentic images
  
    fake/      # Tampered images
  
  test/

    real/      # Authentic images
    
    fake/      # Tampered images

You can use prepare_dataset.py to automatically separate images from:

Au/ → real/

Tp/ → fake/

----------------------------------------------------------------------------------------------------------------------------
🧠 Model Training

Before training, make sure:

- Dataset is correctly placed in dataset/

- Folder model/ exists

      python train_cnn.py


📌 The trained model will be saved as:

    model/model_cnn.h5

----------------------------------------------------------------------------------------------------------------------------
🚀 Future Improvements

- Use Transfer Learning (ResNet50 / EfficientNet)

- Increase dataset size

- Add localization (tampering region detection)

- Deploy to cloud (Heroku / Render / Railway)

----------------------------------------------------------------------------------------------------------------------------
⚠️ Disclaimer

This project is intended for educational and research purposes only.

Results should not be used as legal or forensic evidence without expert validation.

----------------------------------------------------------------------------------------------------------------------------
👨‍💻 Author

Raj
Computer Science / Information Technology

Digital Image Forensics Project

----------------------------------------------------------------------------------------------------------------------------
⭐ Acknowledgements

- CASIA Image Processing Center

- TensorFlow & Keras

- Flask Framework

----------------------------------------------------------------------------------------------------------------------------
📜 License

This repository contains code only.

Dataset license follows CASIA 2.0 terms.
