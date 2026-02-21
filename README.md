# Fruits-360 Image Classification  
**Mini Project – From Pixel to Intelligence: Image Processing and Machine Learning**

## Project Overview
This project focuses on classifying fruit images using image processing techniques and Convolutional Neural Networks (CNN). The model is trained on the Fruits‑360 dataset and demonstrates how raw pixel data can be transformed into intelligent predictions.


## Dataset
**Fruits‑360 Dataset (100×100 images)**  
- Contains thousands of fruit and vegetable images  
- Clean background with consistent lighting  
- RGB images resized to 100×100 pixels  
- Suitable for image processing and deep learning tasks  

Dataset Source:  
https://github.com/fruits-360/fruits-360-100x100  

## Methodology
The project follows these steps:

1. **Data Acquisition** – Load Fruits‑360 dataset  
2. **Preprocessing** – Image resizing, normalization, labeling  
3. **Image Processing** – Histogram analysis and visualization  
4. **Model Building** – CNN architecture using TensorFlow/Keras  
5. **Training** – Model trained on labeled fruit images  
6. **Evaluation** – Accuracy and loss analysis  
7. **Prediction** – Test on unseen fruit images  


<img width="1682" height="898" alt="image" src="https://github.com/user-attachments/assets/09f99005-c72c-4ca7-b094-52583ae2e6bd" />


## CNN Architecture
- Input Layer (100×100×3 RGB Image)  
- Convolution Layer (32 filters, ReLU)  
- Max Pooling  
- Convolution Layer (64 filters, ReLU)  
- Max Pooling  
- Flatten Layer  
- Fully Connected Dense Layer (128 neurons, ReLU)  
- Output Layer (Softmax – Multi-class classification)  

## Results
- The CNN model successfully classified fruit images from the dataset.  
- Training Accuracy: **~97.5%**  
- Loss and accuracy graphs included in the report.  
- Model tested successfully on unseen images.  

## Tools & Technologies
- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy, Matplotlib  
- Google Colab  
- GitHub  

## Model
Trained model saved in Keras format:  
`fruit_classifier.keras`

## Authors
- Sanmati P  
- Sankavi S  
- Sujetha R  

## Conclusion
This project demonstrates how image processing and deep learning can be combined to classify objects from raw pixel data. The CNN model achieved good accuracy and shows the effectiveness of deep learning in image recognition tasks.

## Future Improvements
- Increase dataset size for better accuracy  
- Use advanced architectures (ResNet, MobileNet)  
- Real‑time fruit detection using webcam  
- Hyperparameter tuning for performance improvement  

