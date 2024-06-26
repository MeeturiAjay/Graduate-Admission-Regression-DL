# Graduate-Admission-Regression-DL

Graduate Admission Regression Model using Artificial Neural Networks (ANN) Deep Learning
This project aims to predict graduate admission chances using a regression model based on Artificial Neural Networks (ANN) implemented with TensorFlow.

Overview
The admission process for graduate programs can be highly competitive, and universities often receive a large number of applications. This project seeks to develop a predictive model that can assist in evaluating the likelihood of admission for prospective graduate students. By leveraging machine learning techniques, specifically ANN, the model analyzes various factors such as GRE scores, TOEFL scores, university rankings, and statement of purpose (SOP) scores to predict the probability of admission.

Dataset
The dataset used for training and evaluating the model consists of historical data on graduate admissions, including features such as:

GRE Scores (out of 340)
TOEFL Scores (out of 120)
University Ratings (out of 5)
Statement of Purpose (SOP) Scores (out of 5)
Letter of Recommendation (LOR) Scores (out of 5)
Undergraduate GPA (out of 10)
Research Experience (binary: 0 or 1)


Model Architecture
The ANN model architecture comprises multiple layers of densely connected neurons, with activation functions to introduce non-linearity and regularization techniques to prevent overfitting. The model is trained using gradient descent optimization algorithms to minimize the mean squared error loss function.


Usage
To utilize the model:

Data Preparation: Ensure your dataset is properly formatted and includes the required features for prediction.

Model Training: Train the ANN model using the provided dataset. Tune hyperparameters as necessary to optimize performance.

Model Evaluation: Evaluate the trained model's performance using appropriate metrics such as Mean Absolute Error (MAE) or Mean Squared Error (MSE).

Inference: Use the trained model to make predictions on new data or prospective graduate admission cases.


Requirements
Python 3.x
TensorFlow
Pandas
NumPy
Matplotlib
Jupyter Notebook (optional, for development and visualization)