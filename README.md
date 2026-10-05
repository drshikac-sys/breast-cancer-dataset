# breast-cancer-dataset
# Machine Learning Using Trees

## Course Information

**Course:** CS 4372.501  
**Name:** Drshika Chenna  
**Net ID:** DKC220002  

---

## 1. Project Overview

This project applies four tree-based machine learning methods to the Breast Cancer Wisconsin (Diagnostic) dataset:

1. Decision Tree Classifier
2. Random Forest Classifier
3. AdaBoost Classifier
4. XGBoost Classifier

The objective is to compare the performance of these four models for binary classification and determine which model provides the best overall predictive performance.

The target variable is `diagnosis`, where:

- `B` = Benign
- `M` = Malignant

For modeling, the target was converted to:

- `0` = Benign
- `1` = Malignant

---

## 2. Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset was used for this project.

The dataset contains measurements computed from digitized images of breast mass samples. The target variable indicates whether the tumor is benign or malignant.

The dataset was publicly hosted on UTD Box and loaded directly into the Python notebook.

### Public Dataset URL

https://utdallas.box.com/shared/static/4sxyo3o03s86v8opay2394jhh3znjdcy.csv

The dataset originally contains 569 observations and 33 columns.

During preprocessing:

- The `id` column was removed because it is an identifier and does not provide meaningful predictive information.
- The `Unnamed: 32` column was removed because it contains only missing values.
- The `diagnosis` column was converted from categorical values (`B` and `M`) to numerical values (`0` and `1`).

After preprocessing, the dataset contains 569 observations and 30 numerical predictor variables.

---

## 3. Selected Features

The project did not use all available predictors blindly.

Feature selection was performed using correlations calculated from the training data. The eight predictors selected for the final models were:

1. `concave points_worst`
2. `perimeter_worst`
3. `radius_worst`
4. `concave points_mean`
5. `perimeter_mean`
6. `area_worst`
7. `radius_mean`
8. `area_mean`

These features were selected because they showed strong relationships with the target variable in the training data.

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset into a Pandas DataFrame.
2. Examined the dimensions and structure of the dataset.
3. Checked for missing values.
4. Checked for duplicate observations.
5. Removed the identifier column (`id`).
6. Removed the completely missing `Unnamed: 32` column.
7. Converted the target variable from `B/M` to `0/1`.
8. Examined summary statistics.
9. Examined the distributions and skewness of numerical attributes.
10. Standardized the numerical predictor variables.
11. Normalized the numerical predictor variables.
12. Examined correlations between numerical attributes and the target.
13. Selected the eight most relevant predictors for model construction.

---

## 5. Train-Test Split

The data was divided into training and testing sets using an 80/20 split.

The split was stratified based on the target variable so that the class proportions were preserved.

The following settings were used:

- Training observations: 455
- Testing observations: 114
- Test size: 20%
- Random state: 42
- Stratification: Yes

The random state was fixed to 42 to make the train-test split reproducible.

---

## 6. Models

Four classification models were constructed.

### Decision Tree

A standard `DecisionTreeClassifier` was used.

The model was tuned using GridSearchCV.

Best parameters:

- Criterion: `entropy`
- Maximum depth: `8`
- Minimum samples per leaf: `5`
- Minimum samples required to split: `2`

### Random Forest

A `RandomForestClassifier` was used.

Best parameters:

- Maximum depth: `None`
- Maximum features: `sqrt`
- Minimum samples per leaf: `2`
- Number of estimators: `100`

### AdaBoost

An `AdaBoostClassifier` was used.

Best parameters:

- Learning rate: `0.5`
- Number of estimators: `200`

### XGBoost

An `XGBClassifier` was used.

Best parameters:

- Column subsampling: `1.0`
- Learning rate: `0.1`
- Maximum depth: `4`
- Number of estimators: `100`
- Subsample: `1.0`

---

## 7. Hyperparameter Tuning

GridSearchCV was used to search for suitable hyperparameter values.

A 5-fold Stratified Cross-Validation procedure was used.

The primary scoring metric for hyperparameter selection was **F1-score**.

The use of stratified cross-validation ensures that each fold maintains approximately the same proportion of benign and malignant observations.

---

## 8. Model Evaluation

The final models were evaluated on the test set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

These metrics were used to compare the four models.

---

## 9. Final Results

The final test-set results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 93.86% | 97.30% | 85.71% | 91.14% | 99.52% | 98.99% |
| Random Forest | 95.61% | 100.00% | 88.10% | 93.67% | 99.04% | 98.60% |
| AdaBoost | 94.74% | 95.00% | 90.48% | 92.68% | 99.22% | 98.79% |
| XGBoost | 94.74% | 100.00% | 85.71% | 92.31% | 99.21% | 98.86% |

Based on the final test results, the **Random Forest classifier** performed best overall because it achieved the highest:

- Accuracy: 95.61%
- F1-score: 93.67%

It also achieved 100% precision on the test set.

AdaBoost achieved the highest recall at 90.48%, meaning it correctly identified the largest proportion of malignant cases among the four models.

---

## 10. Confusion Matrices

The test-set confusion matrices were:

### Decision Tree

- True Negatives: 71
- False Positives: 1
- False Negatives: 6
- True Positives: 36

### Random Forest

- True Negatives: 72
- False Positives: 0
- False Negatives: 5
- True Positives: 37

### AdaBoost

- True Negatives: 70
- False Positives: 2
- False Negatives: 4
- True Positives: 38

### XGBoost

- True Negatives: 72
- False Positives: 0
- False Negatives: 6
- True Positives: 36

---

## 11. Feature Importance

Feature importance was examined for the XGBoost model.

The most important features were:

| Feature | Importance |
|---|---:|
| `radius_worst` | 0.6318 |
| `perimeter_worst` | 0.1459 |
| `concave points_mean` | 0.0931 |
| `concave points_worst` | 0.0455 |
| `area_worst` | 0.0266 |
| `area_mean` | 0.0250 |
| `perimeter_mean` | 0.0186 |
| `radius_mean` | 0.0134 |

`radius_worst` had the largest XGBoost feature importance in the final model.

---

# 12. Assumptions

The following assumptions were made during the project:

### Assumption 1: Identifier columns are not useful predictors

The `id` column was treated as an identifier rather than a meaningful feature. Therefore, it was removed before modeling.

### Assumption 2: The completely missing column contains no predictive information

The `Unnamed: 32` column contained only missing values. Because it provided no usable information, it was removed.

### Assumption 3: Diagnosis coding

The original diagnosis labels were assumed to represent:

- `B` = benign
- `M` = malignant

They were therefore encoded as:

- `B → 0`
- `M → 1`

### Assumption 4: Binary classification

The project treats the problem as a binary classification task because there are two possible diagnosis classes.

### Assumption 5: Training-data-based feature selection

Feature selection was performed using the training data rather than the complete dataset. This was done to avoid using information from the test set during model development.

### Assumption 6: Stratified train-test split

An 80/20 train-test split was considered appropriate for evaluating the models. Stratification was used because the dataset contains two classes with different numbers of observations.

### Assumption 7: Fixed random state

A random state of 42 was used to make the train-test split reproducible.

### Assumption 8: F1-score for model selection

F1-score was selected as the primary GridSearchCV scoring metric because it balances precision and recall. This is useful for a medical classification problem where both false positives and false negatives are important.

### Assumption 9: Standardization and normalization

Standardization and normalization were included as part of the preprocessing requirements. Tree-based models generally do not require feature scaling to construct splits, but scaling was performed to satisfy the preprocessing requirements and provide a consistent preprocessing workflow.

### Assumption 10: Test set is used only for final evaluation

The test set was reserved for final model evaluation and was not intended to be used for selecting the final hyperparameters.

### Assumption 11: Model performance is evaluated using the selected test split

The reported accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC values correspond to the final 20% test split using `random_state=42`.

### Assumption 12: Results may vary slightly across environments

Small differences in results may occur if different versions of Python, scikit-learn, XGBoost, NumPy, or other packages are used. The results reported in the project correspond to the environment in which the notebook was executed.

---

# 13. Requirements

The project requires Python 3.x and the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- scipy
- jupyter
- graphviz
- pydot / pydotplus, if required by the tree visualization cells

The easiest way to run the project is through **Google Colab**, because the notebook is already in Jupyter Notebook (`.ipynb`) format.

---

# 14. How to Run the Code in Google Colab

### Step 1: Open Google Colab

Go to:

https://colab.research.google.com/

### Step 2: Upload the notebook

Open the notebook:

`assignment2_cs4372.ipynb`

Upload it to Google Colab.

### Step 3: Install required packages

If the notebook does not already install the required packages, run the package-installation cell before running the model code.

The main additional package required for XGBoost is:

```bash
pip install xgboost
