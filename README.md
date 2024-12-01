
# Brain Tumor Classification using CNN

This deep learning project classifies brain tumors from medical images using a Convolutional Neural Network (CNN). The project includes a user-friendly GUI interface for users to upload MRI images and identify the presence of a tumor. Built using TensorFlow, Keras, OpenCV, and PyQt5 for streamlined image analysis and prediction.

## Prerequisites

Make sure you have Python installed. You'll need to install the following libraries:

```bash
pip install tensorflow keras opencv-python PyQt5 matplotlib
```

## Dataset

Download the **Brain Tumor MRI Dataset** from Kaggle:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/ravi17/brain-mri-images-for-brain-tumor-detection)

1. Create a folder named `dataset` in the project directory.
2. Extract the downloaded dataset and paste it into the `dataset/` directory.

## Steps to Run the Project

### 1. Train the Model

Run the training script to build and save the model:

```bash
python Training_code.py
```

This will train a CNN model on the dataset and save the trained model.

### 2. Run the GUI Interface

Once the model is trained and saved, launch the GUI to classify images:

```bash
python GUI interface code.py
```

This will open the GUI interface where you can upload MRI images.

### 3. Upload Image

Use the GUI to upload an MRI image. The system will predict and display whether the image contains a tumor or not.

## File Structure

- `Training_code.py`: Script to train the CNN model.
- `GUI interfacecode.py`: GUI interface script to upload and classify images.
- `dataset/`: Folder containing the MRI dataset.

## License

This project is open-source and available under the MIT License.
