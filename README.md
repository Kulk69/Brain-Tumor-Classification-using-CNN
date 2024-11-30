# Brain-Tumor-Classification-using-CNN
A deep learning project that classifies brain tumors from medical images using a Convolutional Neural Network (CNN). The project includes a user-friendly GUI interface where users can upload medical images to identify the presence of a tumor. Built with TensorFlow, Keras, and Python for streamlined image analysis and prediction.

Getting Started
Prerequisites
Make sure you have Python installed. You'll need to install the following libraries:
pip install tensorflow keras opencv-python PyQt5 matplotlib  

Extract the dataset and place it in the project directory under data/.

Steps to Run the Project
Train the Model:
Run the training script to build and save the model:

python training_code.py  

Run the GUI Interface:
Once the model is trained and saved, launch the GUI to classify images:

python gui_interface_code.py  
Upload Image:
Use the GUI to upload an MRI image and view the classification result.

File Structure
model.py: CNN model architecture.
gui.py: GUI interface script.
data/: Folder containing the dataset.
