# multidiesase-prediction-using-CNN
**Abstract**:This project aims to develop a Digital Assistance Bot for pharmacy services, utilizing the Rasa Framework. The bot will serve as a personalized assistant to pharmacy customers, handling inquiries regarding medication details, drug interactions, dosage information, and general pharmaceutical guidance. Leveraging advanced natural language processing and machine learning algorithms, it will ensure accurate and contextually relevant responses to user queries. Integration with external databases and APIs will provide access to up-to-date pharmaceutical information, ensuring compliance with industry standards and regulations. The iterative development process will encompass data collection, model training, testing, and refinement to enhance the bot's robustness and adaptability.

By enhancing accessibility and efficiency in pharmacy services, this project seeks to showcase the transformative potential of conversational AI technologies in healthcare delivery. Ultimately, the Digital Assistance Bot aims to promote customer satisfaction, improve overall healthcare experiences, and contribute to the evolution of patient-centered pharmacy services. With its ability to provide personalized assistance and access to accurate information, the bot is poised to revolutionize the way pharmacy interactions take place, offering convenience and reliability to customers while meeting the demands of modern healthcare standards.

**METHODOLOGY** :A proposed model for multi disease prediction using Convolutional Neural Networks 
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
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/db28ef2e-3d3f-45ae-9aa0-5333f8895e97)

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
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/a5d0a6a6-b1e7-4744-bad0-d4cd15023a0d)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/75bdb12b-38de-4dec-a82a-4af4622029b6)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/4322ca3b-d511-4a2a-b0b5-aba3830e152b)

**Models Proposed**
We have used a total of 4 models in the project, Linear Regression, SVM, Random Foreest Classifier, CNN. these models have been indiviualy trained the the performenace metrics of the models have given us the output as the CNN being the best model .


**Website interface development using Streamlit**
Developing a web-based interactive Streamlit application for our Multiple 
Disease Prediction System involves several key steps. Firstly, we design the user 
interface (UI) to enable users to input relevant patient data and visualize predictive 
model outputs. Next, we integrate the predictive models, such as Support Vector 
Machine (SVM) and Logistic Regression, into the Streamlit application to generate 
real- time predictions based on the user inputs. Finally, we deploy the Streamlit 
application to a web server, making it accessible to healthcare practitioners for 
informed decision- making and patient management.
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/86a73ecc-f082-427c-ae2f-ebd384d4090b)

**Results**
the performance metrics used were, accuracy , precison ,recall and F1 score:

Accuracy: While accuracy provides a straightforward measure of overall predictive correctness, it may not be sufficient for assessing model performance in scenarios with imbalanced class distributions. Nonetheless, it offers a fundamental baseline for evaluating the model's predictive prowess by quantifying the proportion of correctly predicted disease outcomes relative to the total predictions made.

Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN) TP: True Positives (correctly predicted positive cases)
TN: True Negatives (correctly predicted negative cases) FP: False Positives (incorrectly predicted positive cases) FN: False Negatives (incorrectly predicted negative cases)

Precision and Recall (Sensitivity): Precision and recall offer complementary insights into the model's ability to correctly identify positive instances while minimizing false positives and negatives, respectively. Precision calculates the ratio of true positive predictions to all positive predictions, emphasizing the model's precision in correctly identifying positive cases. In contrast, recall, also known as sensitivity, measures the proportion of true positive predictions relative to all actual positive instances, elucidating the model's sensitivity in capturing all positive cases, irrespective of false negatives.
Formula: Precision = TP / (TP + FP)
TP: True Positives (correctly predicted positive cases) FP: False Positives (incorrectly predicted positive cases)

Recall: Recall quantifies the model's ability to accurately identify negative instances, indicating its capacity to correctly classify disease-free individuals and mitigate false alarms. It computes the proportion of true negative predictions relative to all actual negative instances, providing crucial insights into the model's ability to maintain high precision in the absence of disease.
Formula: Recall = TP / (TP + FN)

F1 Score: The F1 score represents the harmonic mean of precision and recall, synthesizing both metrics into a single value that balances the trade-off between false positives and false negatives. It serves as a robust measure of a model's overall performance, particularly valuable in scenarios with imbalanced class distributions where accuracy alone may be misleading.

Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/fdf71ec5-485a-486e-ba96-9fb572999fbd)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/41f4fc76-bd95-4853-ad6d-2d4d6cd7a31c)

![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/4774da5b-aaac-4546-a1e8-0703c5450d58)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/3520ea91-25d4-4e8a-81f9-3acf5c4b67fb)

![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/f37e6ca7-dcd6-430d-9bee-40e47c7b640f)
![image](https://github.com/hasil7677/multidiesase-prediction-using-CNN/assets/89244981/a40c8ef5-feee-4ced-a637-c2cdd9bd8ac2)





