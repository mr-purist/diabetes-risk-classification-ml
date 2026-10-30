# Diabetes Risk Classification Using Machine Learning Classifiers

**Course**: Machine Learning (Assignment - 2)  
**Program**: M.Tech (AIML), Work Integrated Learning Programmes Division, BITS Pilani  
**Student Name**: Mayank Gupta  
**BITS Student ID**: 2026AK05256  
**Random State**: 26  

---

## a. Problem Statement

Diabetes mellitus is a chronic metabolic disorder characterized by elevated levels of blood glucose, leading over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. According to global health reports, hundreds of millions of individuals worldwide live with diagnosed or undiagnosed diabetes, with prevalence rising rapidly in low- and middle-income nations.

The objective of this project is to implement, evaluate, and deploy an end-to-end Machine Learning classification framework capable of accurately predicting whether an individual is at risk of diabetes based on clinical diagnostics, physiological measurements, lifestyle habits, and socioeconomic indicators. 

By comparing **5 distinct supervised classification paradigms** (Logistic Regression, Decision Trees, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machines) across **6 comprehensive performance metrics**, this study identifies the optimal predictive architecture and provides an interactive, production-ready Streamlit web application deployed to Streamlit Community Cloud for real-time risk screening.

---

## b. Dataset Description [1 Mark]

* **Source Repository**: Kaggle - [Alex Teboul: Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
* **Underlying Origin**: Centers for Disease Control and Prevention (CDC), Behavioral Risk Factor Surveillance System (BRFSS 2015).
* **Target CSV File**: `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
* **Total Instances (Rows)**: **70,692 survey records** (Exceeds the assignment threshold of $\ge 500$ instances).
* **Total Predictor Features**: **21 features** (Exceeds the assignment threshold of $\ge 12$ features).
* **Class Distribution**: Exactly **50.0% Non-Diabetic (35,346 records)** and **50.0% Diabetic/Prediabetic (35,346 records)**. This curated 50/50 balance ensures that evaluation metrics (Accuracy, ROC-AUC, F1, and MCC) are mathematically unbiased and not inflated by majority-class dominance.
* **Data Preprocessing & Split**: An 80/20 stratified split (`random_state=26`) was applied to generate 56,553 training samples and 14,139 test samples. Continuous and ordinal variables were standardized via `StandardScaler` fitted strictly on training data to prevent lookahead data leakage. A curated balanced slice of 1,500 test samples was exported to `test_data.csv` for Streamlit Cloud evaluation.

### Feature Dictionary

| # | Feature Name | Variable Type | Domain / Range | Description & Clinical Relevance |
| :-: | :--- | :---: | :---: | :--- |
| **1** | `HighBP` | Binary | 0, 1 | 0 = No high blood pressure; 1 = High blood pressure diagnosed |
| **2** | `HighChol` | Binary | 0, 1 | 0 = Normal cholesterol; 1 = High cholesterol diagnosed |
| **3** | `CholCheck` | Binary | 0, 1 | 0 = No cholesterol check in past 5 years; 1 = Cholesterol checked |
| **4** | `BMI` | Continuous | 12 – 98 | Body Mass Index ($	ext{kg}/	ext{m}^2$), primary obesity indicator |
| **5** | `Smoker` | Binary | 0, 1 | Smoked $\ge 100$ cigarettes in entire lifetime (0 = No, 1 = Yes) |
| **6** | `Stroke` | Binary | 0, 1 | Patient history of stroke (0 = No, 1 = Yes) |
| **7** | `HeartDiseaseorAttack` | Binary | 0, 1 | History of coronary heart disease or myocardial infarction (0 = No, 1 = Yes) |
| **8** | `PhysActivity` | Binary | 0, 1 | Physical activity in past 30 days outside routine occupation (0 = No, 1 = Yes) |
| **9** | `Fruits` | Binary | 0, 1 | Consumes fruit $\ge 1$ time per day (0 = No, 1 = Yes) |
| **10** | `Veggies` | Binary | 0, 1 | Consumes vegetables $\ge 1$ time per day (0 = No, 1 = Yes) |
| **11** | `HvyAlcoholConsump` | Binary | 0, 1 | Heavy drinking: Adult men $>14$ drinks/week, women $>7$ drinks/week |
| **12** | `AnyHealthcare` | Binary | 0, 1 | Has healthcare coverage including health insurance / HMO (0 = No, 1 = Yes) |
| **13** | `NoDocbcCost` | Binary | 0, 1 | Needed to visit physician in past 12 months but prevented by cost (0 = No, 1 = Yes) |
| **14** | `GenHlth` | Ordinal | 1 – 5 | General health rating (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor) |
| **15** | `MentHlth` | Discrete | 0 – 30 | Days of poor mental health experienced in past 30 days |
| **16** | `PhysHlth` | Discrete | 0 – 30 | Days of physical illness or injury experienced in past 30 days |
| **17** | `DiffWalk` | Binary | 0, 1 | Serious difficulty walking or climbing stairs (0 = No, 1 = Yes) |
| **18** | `Sex` | Binary | 0, 1 | Biological sex (0 = Female, 1 = Male) |
| **19** | `Age` | Ordinal | 1 – 13 | 13-level age category (1 = 18–24, 5 = 40–44, 9 = 60–64, 13 = 80+) |
| **20** | `Education` | Ordinal | 1 – 6 | Education level completed (1 = None/Kindergarten, 4 = High School, 6 = College 4+ yrs) |
| **21** | `Income` | Ordinal | 1 – 8 | Household income scale (1 = $<\$10	ext{k}$, 5 = $\$25	ext{k}	ext{–}\$35	ext{k}$, 8 = $\ge \$75	ext{k}$) |
| **T** | **`Diabetes_binary`** | **Binary Target** | **0, 1** | **0 = Non-Diabetic; 1 = Prediabetic / Diabetic** |

---

## c. GitHub Repository Link [1 Mark]

* **Public GitHub Repository**: [https://github.com/mr-purist/diabetes-risk-classification-ml](https://github.com/mr-purist/diabetes-risk-classification-ml)

The repository contains:
* `app.py`: Complete interactive Streamlit frontend application.
* `requirements.txt`: Specified Python library dependencies.
* `README.md`: Exhaustive project documentation and evaluation report.
* `test_data.csv`: Curated 1,500-instance balanced test partition.
* `train_models.py`: Full end-to-end model training, validation, and evaluation script.
* `model/`: Serialized pre-trained `.joblib` artifacts for all 5 classifiers and the preprocessor pipeline.

---

## d. Models Used & Comparison Table [5 Marks]

All 5 classification models were trained on the identical standardized training set (56,553 rows) and evaluated on the full unseen test set (14,139 rows) using `random_state=26`. The 6 mandatory evaluation metrics are compiled in the comparison table below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7443 | 0.8242 | 0.7365 | 0.7608 | 0.7485 | 0.4889 |
| **Decision Tree** | 0.7382 | 0.8115 | 0.7323 | 0.7511 | 0.7416 | 0.4766 |
| **KNN** | 0.7288 | 0.7938 | 0.7122 | 0.7679 | 0.7390 | 0.4589 |
| **Naive Bayes** | 0.7211 | 0.7859 | 0.7254 | 0.7115 | 0.7184 | 0.4422 |
| **Support Vector Machine** | **0.7499** | 0.8202 | 0.7212 | **0.8150** | **0.7652** | **0.5041** |

---

## Observations on Model Performance [3 Marks]

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest ROC-AUC score (**0.8242**) and highest Precision (**0.7365**) among all models. With numerical features standardized via `StandardScaler`, log-odds parameter estimation converged efficiently in under 1 second. The model establishes a robust, highly calibrated linear decision boundary that balances specificity and sensitivity with minimal variance. |
| **Decision Tree** | Delivered strong competitive performance (**73.82% Accuracy**, **0.8115 AUC**, **0.7416 F1**). Controlled tree regularization (`max_depth=6`, `min_samples_leaf=10`) prevented overfitting. Feature importance analysis revealed that top splits isolate `GenHlth` (General Health), `HighBP` (High Blood Pressure), and `BMI`, aligning with established clinical diagnostics while offering unmatched human interpretability. |
| **KNN** | Achieved **72.88% Accuracy**, **0.7938 AUC**, and an impressive **76.79% Recall**. With $k=9$ neighbors using Euclidean distance, KNN effectively identified local topological risk clusters in the 21-dimensional space. However, test inference time was comparatively higher ($O(N \cdot D)$), and performance was moderately constrained by the curse of dimensionality across 21 features. |
| **Naive Bayes** | Yielded **72.11% Accuracy**, **0.7859 AUC**, and **0.4422 MCC**. Despite the strong assumption of conditional independence among predictor attributes—which is violated by naturally correlated clinical factors such as `HighBP`, `HighChol`, and `BMI`—Gaussian Naive Bayes provided an exceptionally fast, parameter-free probabilistic baseline without iterative gradient descent. |
| **Support Vector Machine** | Attained the highest overall Accuracy (**74.99%**), highest Recall (**81.50%**), highest F1 Score (**0.7652**), and highest Matthews Correlation Coefficient (**0.5041**). The non-linear Radial Basis Function (RBF) kernel successfully projected the 21 clinical attributes into a higher-dimensional space, resolving non-linear risk interactions where linear hyperplanes hit a performance ceiling. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVC with RBF Kernel)** is the unambiguous winner for this diabetes screening task.<br><br>**Clinical & Statistical Justification**:<br>1. **Top Composite Quality**: SVM leads all models across Accuracy (74.99%), F1 Score (0.7652), and Matthews Correlation Coefficient (0.5041). MCC is the gold-standard metric for binary classification because it incorporates all 4 quadrants ($TP, TN, FP, FN$) and is immune to threshold artifacts.<br>2. **Clinical Imperative (High Sensitivity / Recall)**: In medical screening, a False Negative (failing to identify an individual with diabetes) carries severe consequences, potentially leading to untreated complications such as neuropathy, retinopathy, or cardiovascular failure. SVM achieves an outstanding **81.50% Recall**, capturing $>8$ out of every 10 diabetic patients—the highest sensitivity of any model evaluated.<br>3. **Non-Linear Etiology**: Diabetes pathology arises from intricate, non-linear interactions among metabolic, lifestyle, and demographic risk factors. The RBF kernel captures these complex multivariate boundaries more effectively than linear baselines. |

---

## Streamlit Application Features

The deployed Streamlit web application encompasses the following features as per the assignment guidelines:
1. **Dataset Upload Option (CSV)**: Users can upload custom test CSV datasets containing the 21 input features and target. If no file is uploaded, the app automatically falls back to the bundled 1,500-instance `test_data.csv`. The app evaluates saved models directly without retraining.
2. **Model Selection Dropdown**: Allows seamless toggling between individual classifiers (SVM, Logistic Regression, Decision Tree, KNN, Naive Bayes) as well as an interactive "Compare All 5 Models" master view.
3. **Display of Evaluation Metrics**: Renders all 6 metrics (Accuracy, ROC-AUC, Precision, Recall, F1-Score, MCC) using styled metric cards.
4. **Confusion Matrix & Classification Report**: Renders an annotated Seaborn heatmap of the confusion matrix alongside detailed per-class classification reports and ROC characteristic curves.
