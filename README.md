# What-is-on-My-Skin
---

<p align = "center">
    <img src = "https://github.com/user-attachments/assets/e4390186-4f22-48b4-9757-ce1a33562477">
</p>

<br>

## I. Project Name and Description

"What's on My Skin" is an **AI algorithm-based** facial skin disease prediction and treatment suggestion system. 

<br>

I've developed a system that allows patients with skin conditions to upload a facial photo and determine whether their skin is normal(정상 피부), has acne(여드름), acne scars(여드름 흉터), or hyperpigmentation(색소침착).

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
<img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white"> 
<img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"> 
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white">

## IV. System Usage Example

![ex1](https://github.com/user-attachments/assets/a908bb23-7ff9-42d3-b92b-6726ee31bbb9)

<br>

![ex2](https://github.com/user-attachments/assets/f30ab160-01e7-4c70-979d-ac9588464298)


## V. Installation(for Testers)



## VI. Key Development Considerations (AI Focus)
 - I **manually collected** 250 facial images for each of the three skin diseases as well as normal skin.
 - Due to the limitations in the dataset size that an individual can gather, data augmentation techniques were applied. 
 - The CNN algorithm used was a **pretrained DenseNet121.**

 
 > A document summarizing all the things considered during project development is located in **project_design.pdf in the results folder.**

