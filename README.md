



Multiple Disease Prediction System using Machine Learning: This project provides a streamlit web application for predicting multiple diseases, including diabetes, Parkinson's disease, and heart disease, using machine learning algorithms. The prediction models are deployed using Streamlit, a Python library for building interactive web applications.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)


## Introduction

The Multiple Disease Prediction project aims to create a user-friendly web application that allows users to input relevant medical information and receive predictions for different diseases. The machine learning models trained on disease-specific datasets enable accurate predictions for diabetes, Parkinson's disease, and heart disease.

## Features

The Multiple Disease Prediction web application offers the following features:

- **User Input**: Users can input their medical information, including age, gender, blood pressure, cholesterol levels, and other relevant factors.
- **Disease Prediction**: The application utilizes machine learning models to predict the likelihood of having diabetes, Parkinson's disease, and heart disease based on the inputted medical data.
- **Prediction Results**: The predicted disease outcomes are displayed to the user, providing an indication of the probability of each disease.
- **Visualization**: Visualizations are generated to highlight important features and provide insights into the prediction process.
- **User-Friendly Interface**: The web application offers an intuitive and user-friendly interface, making it easy for individuals without technical knowledge to use the prediction tool.

## Setup

To use this project locally, follow these steps:

1. Clone the repository:
   
```bash
git clone https://github.com/SagarMandal7/Multiple-Disease-Prediction-System-using-Machine-Learning/tree/main
```

2. Install the required dependencies by running:
   
```bash
pip install -r requirements.txt
```

3. Download the pre-trained machine learning models for diabetes, Parkinson's disease, and heart disease. Make sure to place them in the appropriate directories within the project structure.

4. Update the necessary configurations and file paths in the project files.

## Usage

To run the Multiple Disease Prediction web application, follow these steps:

1. Open a terminal or command prompt and navigate to the project directory.

2. Run the following command to start the Streamlit application:

```bash
streamlit run multiplediseaseprediction.py
```

3. Access the web application by opening the provided URL in your web browser.

4. Input the relevant medical information as requested by the application.

5. Click the "Predict" button to generate predictions for diabetes, Parkinson's disease, and heart disease based on the provided data.

6. View the prediction results and any accompanying visualizations or insights.



**METHODOLOGY **
A proposed model for multi disease prediction using Convolutional Neural Networks 
(CNNs) involves several key steps. Firstly, collect and preprocess a diverse dataset of 
medical images labeled with multiple diseases. Then, design a CNN architecture 
comprising convolutional, pooling, dropout, and fully connected layers, culminating 
in an output layer with softmax activation for multi-class prediction. Train the model 
on a split dataset, optimizing performance using techniques like learning rate 
scheduling and early stopping, while monitoring metrics such as accuracy and loss. 
Evaluate the trained model's performance on a separate testing set, utilizing metrics 
like accuracy, precision, recall, and F1-score across various diseases. Deploy the 
model in a production environment, ensuring compliance with healthcare regulations 
and integrating it with existing systems. Continuously improve the model through 
feedback, fine-tuning, and updates to enhance its accuracy and clinical relevance, 
fostering collaboration with domain experts throughout the process.
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/56dadc8f-2ca5-410d-aa9b-ad70fb1f1a16)

**Data Set** : 
The dataset selection process for our Multiple Disease Prediction System is 
crucial, as it forms the cornerstone of our predictive model's accuracy and reliability. 
We meticulously curate a comprehensive dataset that encompasses a wide range of 
medical data types, including patient demographics, clinical history, laboratory test 
results, imaging scans, and genetic information. Ensuring data quality and 
representativeness is paramount; thus, we conduct rigorous data cleaning, 
normalization, and preprocessing to remove noise, handle missing values, and 
standardize feature formats. Additionally, we employ stratified sampling techniques 
to address class imbalance issues, ensuring adequate representation of each disease 
category in our dataset. Collaborating with healthcare institutions and leveraging 
publicly available datasets, we aim to construct a diverse and inclusive dataset that 
captures the heterogeneity of disease manifestations and patient populations. 
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/3f622790-517d-4335-91c9-3ee961ccb960)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/248881e0-af7d-47fb-98b4-82687903c09d)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/759063f0-e079-4ad4-a68c-65362363c176)
**Website interface development using Flask**
Developing a web-based interactive Streamlit application for our Multiple 
Disease Prediction System involves several key steps. Firstly, we design the user 
interface (UI) to enable users to input relevant patient data and visualize predictive 
model outputs. Next, we integrate the predictive models, such as Support Vector 
Machine (SVM) and Logistic Regression, into the Streamlit application to generate 
real- time predictions based on the user inputs. Finally, we deploy the Streamlit 
application to a web server, making it accessible to healthcare practitioners for 
informed decision- making and patient management.![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/328e4b7e-5726-4f25-80fb-c6e5c2853250)


**Accuracy**: While accuracy provides a straightforward measure of overall predictive correctness, it may not be sufficient for assessing model performance in scenarios with imbalanced class distributions. Nonetheless, it offers a fundamental baseline for evaluating the model's predictive prowess by quantifying the proportion of correctly predicted disease outcomes relative to the total predictions made.

**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN) TP: True Positives (correctly predicted positive cases)
TN: True Negatives (correctly predicted negative cases) FP: False Positives (incorrectly predicted positive cases) FN: False Negatives (incorrectly predicted negative cases)

**Precision and Recall** (Sensitivity): Precision and recall offer complementary insights into the model's ability to correctly identify positive instances while minimizing false positives and negatives, respectively. Precision calculates the ratio of true positive predictions to all positive predictions, emphasizing the model's precision in correctly identifying positive cases. In contrast, recall, also known as sensitivity, measures the proportion of true positive predictions relative to all actual positive instances, elucidating the model's sensitivity in capturing all positive cases, irrespective of false negatives.
Formula: Precision = TP / (TP + FP)
TP: True Positives (correctly predicted positive cases) FP: False Positives (incorrectly predicted positive cases)

Recall: Recall quantifies the model's ability to accurately identify negative instances, indicating its capacity to correctly classify disease-free individuals and mitigate false alarms. It computes the proportion of true negative predictions relative to all actual negative instances, providing crucial insights into the model's ability to maintain high precision in the absence of disease.
Formula: Recall = TP / (TP + FN)

**F1 Score:** The F1 score represents the harmonic mean of precision and recall, synthesizing both metrics into a single value that balances the trade-off between false positives and false negatives. It serves as a robust measure of a model's overall performance, particularly valuable in scenarios with imbalanced class distributions where accuracy alone may be misleading.

Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
**Result tables and graphs**
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/cf9fb6f1-8efe-4561-9bb0-9d7a7bbce020)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/d0454eb3-516f-49a9-b604-b33ad0844731)

![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/672cf66a-57a3-487d-bba9-9ae5e0fde0c6)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/74026973-0db5-4b61-82b5-7fe0df29c5c2)


![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/35242e5d-6554-43b6-9dd8-41f4cac7e7da)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/b242ac3e-8d1a-4cf7-8da7-8f55709cefb8)

