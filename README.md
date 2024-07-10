# Breast Cancer Detection using Neural Networks

This repository contains code for building and training a neural network model to detect breast cancer using a dataset provided by an instructor during an internship. The dataset might be the UCI Breast Cancer Wisconsin (Diagnostic) dataset, which is commonly used for such tasks. The model uses a deep learning approach with a grid search for hyperparameter tuning to improve the accuracy of breast cancer detection.

## Dataset

The dataset used for this project was provided by an instructor during an internship. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It includes the following columns:

- `id`
- `diagnosis`
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `symmetry_mean`
- `fractal_dimension_mean`
- `radius_se`
- `texture_se`
- `perimeter_se`
- `area_se`
- `smoothness_se`
- `compactness_se`
- `concavity_se`
- `concave points_se`
- `symmetry_se`
- `fractal_dimension_se`
- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave points_worst`
- `symmetry_worst`
- `fractal_dimension_worst`

## Dependencies

The following Python libraries are required to run the code:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- keras

## You can install these libraries using `pip`:

pip install numpy pandas matplotlib seaborn scikit-learn keras


## Data Preprocessing

-Load the dataset.
-Encode the target variable (diagnosis) using LabelEncoder.
-Drop irrelevant columns (radius_worst, perimeter_worst, area_worst, concave points_worst, perimeter_se, id).
-Check for missing values.
-Visualize the correlation matrix using a heatmap.
-Normalize the feature variables using MinMaxScaler.
-Split the dataset into training and testing sets.

## Model Building

-Define a function build_classifier to create a Sequential model with:
-Input layer with 16 units and ReLU activation
-Hidden layer with 16 units and ReLU activation
-Output layer with 1 unit and sigmoid activation
-Compile the model using Adam optimizer and binary crossentropy loss.
-Use GridSearchCV for hyperparameter tuning (batch_size and epochs).
-Train the model using the best parameters obtained from GridSearchCV.
-Evaluate the model's performance using accuracy and loss metrics.

## Model Evaluation

-Plot the training and validation accuracy.
-Plot the training and validation loss.
-Predict the test set results.
-Convert probabilities into binary predictions.
-Compute the confusion matrix and accuracy score.
-Visualize the confusion matrix using a heatmap.

## Results
-The model's performance is evaluated using the confusion matrix and accuracy score. The final accuracy score is printed, and the confusion matrix is visualized using a heatmap.
-Confusion Matrix:
array([[66,  1],
       [ 2, 45]])
-Accuracy Score: 97.37%

## Usage

-Clone the repository: 
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection
-Run the script:python breast_cancer_detection.py

## License
This project is licensed under the MIT License. Free to use.
