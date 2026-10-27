# Wine Classification: Machine Learning Model Comparison

## Project Overview

This project compares six machine learning classification algorithms using the Wine dataset provided by Scikit-learn. The objective is to predict wine classes from 13 chemical features and evaluate model performance using multiple classification metrics.

## Objectives

- Explore and understand the Wine dataset through exploratory data analysis (EDA).
- Prepare the data using a stratified train-test split and feature scaling.
- Train six machine learning classification models.
- Evaluate each model using accuracy, precision, recall, F1-score, and confusion matrices.
- Compare model performance and identify the best-performing classifier.

## Dataset

The dataset was loaded using `load_wine()` from `sklearn.datasets`.

- **Samples:** 178
- **Features:** 13 numerical chemical measurements
- **Target classes:** 3
- **Missing values:** None detected in the input features

The target represents three wine classes, not wine quality ratings.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook / Google Colab

## Methodology

1. **Data loading:** Loaded the Wine dataset and converted the features into a Pandas DataFrame.
2. **Exploratory data analysis:** Examined the dataset structure, descriptive statistics, and target-class distribution.
3. **Data preparation:** Separated the input features from the target variable and split the data into 80% training and 20% testing sets using stratified sampling.
4. **Feature scaling:** Applied `StandardScaler` to the training and testing features for Logistic Regression, KNN, and SVM. The scaler was fitted on training data only.
5. **Model training:** Trained six classification algorithms.
6. **Model evaluation:** Calculated accuracy, precision, recall, F1-score, and confusion matrices.
7. **Model comparison:** Ranked the classifiers according to their test accuracy.

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbours (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

## Results

| Model | Test Accuracy |
|---|---:|
| Random Forest | 100.00% |
| Logistic Regression | 97.22% |
| K-Nearest Neighbours (KNN) | 97.22% |
| Gaussian Naive Bayes | 97.22% |
| Support Vector Machine (SVM) | 97.22% |
| Decision Tree | 94.44% |

Random Forest achieved the highest test accuracy, correctly classifying all 36 test samples. Logistic Regression, KNN, Gaussian Naive Bayes, and SVM each achieved 97.22%, while Decision Tree achieved 94.44%.

These results apply to the selected train-test split. Cross-validation would help assess whether the models perform consistently across different data splits.

## Key Findings

- All six models achieved test accuracy above 94%.
- Random Forest performed best on the test set, with 100% accuracy.
- Four models achieved the same accuracy of 97.22%, despite differences in their class-level precision and recall.
- Feature scaling was applied to the scale-sensitive models to improve the fairness and effectiveness of their training.

## Conclusion

This project demonstrates the application of supervised machine learning to a multiclass classification problem. Random Forest was the best-performing model on the test set, although further validation is needed to assess its generalization to unseen data.

## Repository Contents

- `wine_classification.ipynb` — Notebook containing the code, analysis, evaluation metrics, and interpretations.
- `README.md` — Project overview, methodology, results, and conclusion.
- `LICENSE` — Project license, if included.

## How to Run

1. Clone or download this repository.
2. Install the required libraries.
3. Open `wine_classification.ipynb` in Jupyter Notebook or Google Colab.
4. Run the notebook cells in order.

Install the required libraries with:

```bash
pip install pandas numpy scikit-learn jupyter
```

## License

This project is distributed under the MIT License, if the repository includes an MIT `LICENSE` file.
