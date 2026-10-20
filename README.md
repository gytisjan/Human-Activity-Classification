# Human Activity Classification Using Machine Learning

A 2024 project for the **Machine Learning Methods (P160B124)** course at Kaunas University of Technology, completed together with **Matas Masalskis**. The project investigated multiclass human activity recognition using smartphone sensor features.

This repository presents **my own code contribution**: feature exploration, correlation-based feature selection, Random Forest, and Naive Bayes. XGBoost and Nearest Shrunken Centroids were investigated by my teammate but are not implemented here.

## Dataset

The data correspond to the [Smartphone-Based Recognition of Human Activities and Postural Transitions](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) dataset from the UCI Machine Learning Repository. Accelerometer and gyroscope recordings collected at 50 Hz were processed into time-domain and frequency-domain features. The dataset includes six basic activities and six postural transitions.

| Property | Value |
|---|---:|
| Training observations | 7,767 |
| Test observations | 3,162 |
| Numeric predictors | 561 |
| Activity classes | 12 |
| Missing values in the supplied CSV files | 0 |

The activity classes are imbalanced: classes 7–12 have considerably fewer observations. Feature selection removes predictors with high pairwise correlation using a **0.8 threshold**; the original report retained **176 predictors** (177 columns including the target).

Source: Reyes-Ortiz, J., Anguita, D., Oneto, L., & Parra, X. (2015). *Smartphone-Based Recognition of Human Activities and Postural Transitions*. UCI Machine Learning Repository. https://doi.org/10.24432/C54G7M (CC BY 4.0).

## Methods and results

The R script visualizes selected features and their correlations, performs correlation-based feature removal, and evaluates:

- **Random Forest:** an initial 2-tree experiment, a 500-tree model, and comparisons of different `mtry` values using out-of-bag error and class-wise accuracy.
- **Naive Bayes:** default settings, Laplace smoothing experiments, and kernel-density estimation for continuous features.

The original 2024 project report records the following test accuracies:

| Method | Reported test accuracy |
|---|---:|
| Random Forest | **93.90%** |
| Naive Bayes | **82.89%** |

For Random Forest, the report lists **93.90% for 400 trees** and **93.61% for 300 trees**. Its text nevertheless recommends 300 trees, so both outcomes are preserved here rather than silently resolving the discrepancy. Classification was generally more difficult for the less frequent activity classes.

These values are **historical results from the team report**, not newly reproduced R outputs. The cleaned original R script corrects technical issues, including the class label type, exclusion of the target from feature selection, and removal of an invalid two-class prior specification for the 12-class Naive Bayes task. R results may therefore differ from the report.

## Running the project

1. Install R and the required packages:

   ```r
   install.packages(c("caret", "randomForest", "e1071", "ggcorrplot"))
   ```

2. Place `activity_train.csv` and `activity_test.csv` in a local `data/` directory:

   ```text
   human-activity-classification/
   ├── Machine_Learning.R
   ├── README.md
   ├── .gitignore
   └── data/
       ├── activity_train.csv
       └── activity_test.csv
   ```

3. From the project root in R or RStudio, run:

   ```r
   source("Machine_Learning.R")
   ```

The script prints the selected feature count and classification results and displays exploratory plots, OOB error curves, and class-wise accuracy charts. The original project CSV files are not included in the repository because of their size; the UCI dataset is linked above. The UCI download uses separate feature matrices and labels, so its files must be combined into `activity_train.csv` and `activity_test.csv` (with the `activity` column) to run this script. The supplied project CSV files have the expected 561 numeric columns plus `activity`, and train/test headers match.

## Contributions

- **Gytis Jančaitis:** feature-distribution exploration, correlation-based feature removal, Random Forest, and Naive Bayes.
- **Matas Masalskis:** class-frequency exploration, variable importance, XGBoost, and Nearest Shrunken Centroids.

The original report was prepared jointly. This repository contains only the portions of the R implementation attributed to Gytis. The code is based on the original R script written for the project, with corrections and clearer comments. End-to-end execution in R remains to be verified.
