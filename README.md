Dataset
Download the Brain Tumor MRI Dataset from Kaggle: Brain Tumor MRI Dataset

Create a folder named dataset in the project directory.
Extract the downloaded dataset and paste it into the dataset/ directory.
Steps to Run the Project
1. Train the Model
Run the training script to build and save the model:

bash
Copy code
python training_code.py
This will train a CNN model on the dataset and save the trained model.

2. Run the GUI Interface
Once the model is trained and saved, launch the GUI to classify images:

bash
Copy code
python GUI interface code.py
This will open the GUI interface where you can upload MRI images.

3. Upload Image
Use the GUI to upload an MRI image. The system will predict and display whether the image contains a tumor or not.

File Structure
training_code.py: Script to train the CNN model.
GUI interface code.py: GUI interface script to upload and classify images.
dataset/: Folder containing the MRI dataset.
