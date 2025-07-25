# 😷 Face Mask Detection using OpenCV & Deep Learning

This project is a real-time Face Mask Detection system using **OpenCV**, **TensorFlow/Keras**, and **Haar Cascade** for face detection. It uses a deep learning model to classify whether a person is wearing a mask or not via webcam feed.

---

## 📁 Project Structure

<pre>
face-mask-detector/
├── detect_mask_video.py 
├── mask_detector.h5 # Pre-trained Keras model (H5 format)
├── requirements.txt # Required Python packages
├── face_detector/
│ └── haarcascade_frontalface_default.xml 
├── venv/ # Virtual environment
</pre>


---

## 🧠 How It Works

1. **Face Detection:**  
   Uses OpenCV’s `haarcascade_frontalface_default.xml` to detect faces in video frames.

2. **Mask Classification:**  
   Each detected face is preprocessed and passed to a trained CNN model (`mask_detector.h5`) which predicts:
   - Mask 😷 (Spectacles are considered as a mask)
   - No Mask 😐

3. **Output Display:**  
   Draws bounding boxes and labels around faces in real-time webcam feed.

---

## 🧰 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Typical contents of requirements.txt:  
- opencv-python  
- tensorflow  
- keras  
- numpy  

>You may also want to create and activate a virtual environment before installing.


## 🚀 Run the Project
```bash
python detect_mask_video.py
```

This will:  
- Open your default webcam  
- Detect faces  
- Classify each face as "Mask" or "No Mask"  
- Display results live on screen

Press Q to quit the video window.  

## 🔒 Model Details  
- File: mask_detector.h5  
- Format: Keras H5 (TensorFlow compatible)  
- Input shape: 224x224 RGB images  
- Output: Binary classification (Mask / No Mask)  

## 📦 Notes
- Make sure the file `haarcascade_frontalface_default.xml` is placed inside `face_detector/`
- This project doesn't use any paid APIs — it's completely local and open-source.

## ❌ Known Issues
- Webcam feed might not stop if the video window is closed forcefully. Use Q to quit cleanly.

## 📸 Sample Output
![Masked Photo](1.jpg)  

![Unmasked Photo](2.jpg)  


## 🙌 Credits
- Face detection: OpenCV Haar Cascades  
- Mask classification: Custom trained Keras model  
- Inspiration: COVID-19 real-time prevention systems

## 📌 License
This project is for educational and research purposes only. Feel free to fork and extend!

## Author
Pavanchandra Devang L  
📧 pavanchandradevangl@gmail.com  
🔗 [LinkedIn - Pavanchandra Devang L](https://www.linkedin.com/in/pavanchandra-devang-l-01038616a/)