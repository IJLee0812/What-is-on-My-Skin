# 🔍 What's on My Skin?
---

<p align = "center">
    <img src = "https://github.com/user-attachments/assets/e4390186-4f22-48b4-9757-ce1a33562477">
</p>

<br>

## I. Project Name and Description

"What's on My Skin?" is an **AI algorithm-based** facial skin disease prediction and treatment suggestion system. 

<br>

I've developed a system that allows patients with skin conditions to upload a facial photo and determine whether their skin is normal, has acne, acne scars, or hyperpigmentation.

<br>

After analyzing the patient’s skin condition, the system provides the analysis results along with solutions for each specific skin disease.

## II. Project Duration and Milestones

- 24.07.15 ~ 24.07.28: Investigated detailed classification of each symptoms and identified solutions; gathered facial skin disease datasets for each symptom.

- 24.07.29 ~ 24.07.31: Labeled datasets, performed data preprocessing and augmentation.

- 24.08.02 ~ 24.08.15: Conducted model development and training, followed by testing and performance evaluation to optimize the model.

- 24.08.16 ~ 24.08.22: Developed a demo version of the webservice.

## III. Technology Stacks Used

### LANGUAGE
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 

### AI MODEL
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/scikit&ndash;learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">

### WEBSERVICE
<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white">
<img src="https://img.shields.io/badge/css3-1572B6?style=for-the-badge&logo=css3&logoColor=white"> 
<img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"> 
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white">

## IV. System Usage Example

![ex1](https://github.com/user-attachments/assets/a908bb23-7ff9-42d3-b92b-6726ee31bbb9)

<br>

![ex2](https://github.com/user-attachments/assets/f30ab160-01e7-4c70-979d-ac9588464298)


## V. Installation (for Webservice Testers)

1. Clone the repository :
```plaintext
git clone https://github.com/IJLee0812/What-is-on-My-Skin.git
```

2. Start the web server : 
```plaintext
cd What-is-on-My-Skin/webservice

pip install Flask Pillow torch torchvision -> Commands to install the tools and libraries required to run app.py

python3 app.py
```

3. Open the webservice in your browser : 
```plaintext
http://127.0.0.1:5000 or http://localhost:5000
```

## VI. Key Development Explanation (AI)

### 1. About Data
 - I **manually collected** 250 facial images for each of the three skin diseases as well as normal skin. Below are the four class names and Korean translations that the AI ​​model must classify.
    - acne(여드름)
    - acne_scar(여드름 흉터)
    - hyperpigmentation(색소침착)
    - normal(정상 피부)

<br>

 - Due to the limitations in the dataset size that an individual can gather, data augmentation and 5-Fold Validation techniques were applied. 
  
### 2. Augmentation Process

    CenterCrop(224) -> RandomHorizontalFlip() -> RandomRotation(degrees = 20) -> ToTensor() -> Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

- CenterCrop(224): This operation resizes the input image by cropping it to 224x224 pixels from the center. This ensures that the key features of the image are preserved.

- RandomHorizontalFlip(): The image is randomly flipped horizontally with a 50% probability. This technique helps the model generalize better by introducing variations in the orientation of the images.

- RandomRotation(degrees = 20): The image is rotated randomly within a range of ±20 degrees. This augmentation technique is useful for making the model invariant to slight rotations in the input images.

- ToTensor(): This transforms the image into a PyTorch tensor, where the pixel values are scaled to a range between 0 and 1. Tensors are the primary data structure used in PyTorch models.

- Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]): The image tensor is normalized using a mean of [0.485, 0.456, 0.406] and a standard deviation of [0.229, 0.224, 0.225]. These values are standard for models pretrained on ImageNet, helping to standardize the input data.

### 3. AI Training

 - The **CNN** algorithm used was a **pretrained DenseNet121.** Below is a schematic diagram of the structure of DenseNet.

![DenseNet121](https://github.com/user-attachments/assets/a3c13fa8-83a2-4084-891a-2fd75f04247c)
 

- To advance training, Early Stopping and LR scheduling techniques were applied.
  - LR Scheduler : ReduceLROnPlateau 

### 4. Model Performance Evaluation

<p align = "center">
    <img src = "https://github.com/user-attachments/assets/d7248979-8ff7-4972-a4ce-30f887d339ee">
</p>

<br>

<p align = "center">
    <img src = "https://github.com/user-attachments/assets/fcfbf062-b8b0-4128-9f9b-bab09c85624c">
</p>

<br>

> The model achieved meaningful level of accuracy in both training and validation (around 88%), indicating strong performance with minimal overfitting. <br><br> The training process effectively increased accuracy, suggesting that learning has largely stabilized, and further epochs may not yield significant improvements.

<br>

<p align = "center">
    <img src = "https://github.com/user-attachments/assets/aca09be7-611e-453a-ab1b-064a14fb7367">
</p>

<br>

<p align = "center">
    <img src = "https://github.com/user-attachments/assets/477ce3cf-a8e2-4276-9d53-7b59c70f6293">
</p>

 ---

