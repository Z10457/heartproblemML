# Heart Disease Prediction with k-NN

Predicting the presence of heart disease from clinical measurements using a k-nearest-neighbors classifier, built in Python with scikit-learn.

**Result:** 87.0% accuracy and 90.5% recall on a held-out test set of 138 patients. Always predicting "heart disease" would score about 54% on the same set.

## Dataset

- Kaggle *Heart Failure Prediction Dataset* ([add link](https://www.kaggle.com/)), file `heart.csv`
- 918 patients, 11 clinical features plus the target `HeartDisease`
- Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- About 55% of patients have heart disease, and about 79% of patients are male

## Approach

1. **Cleaning:** removed one record with a resting blood pressure of 0. Replaced 172 cholesterol values recorded as 0 (not physiologically possible) with the median cholesterol of patients with the same diagnosis (see Limitations).
2. **Exploration:** count plots of every categorical feature split by diagnosis, plus a correlation heatmap.
3. **Encoding:** one-hot encoded the categorical columns.
4. **Features:** selected six predictors: `MaxHR`, `Oldpeak`, `Sex_M`, `ExerciseAngina_Y`, `ST_Slope_Flat`, `ST_Slope_Up`.
5. **Scaling:** `MinMaxScaler`, fit on the training data only, so k-NN distances aren't dominated by large-valued features.
6. **Model:** k-NN classifier. Tuned the number of neighbors (1 to 19) and the distance metric (Minkowski vs. Manhattan) with 5-fold `GridSearchCV`. Best parameters: k = [fill in], metric = [fill in], cross-validated accuracy = [fill in].
7. **Evaluation:** 85/15 train/test split, evaluated once on the held-out test set.

## Results

| Metric | Value |
|---|---|
| Accuracy | 87.0% (120 / 138) |
| Recall (sensitivity) | 90.5% (67 / 74) |
| Precision | 85.9% (67 / 78) |
| Specificity | 82.8% (53 / 64) |

Confusion matrix on the test set:

| | Predicted: no disease | Predicted: disease |
|---|---|---|
| **Actual: no disease** | 53 | 11 |
| **Actual: disease** | 7 | 67 |

Recall is the metric I care most about here. In a screening setting, missing a patient with heart disease (7 of 74 here) costs more than a false alarm (11 of 64).

## Limitations

- **Small test set:** with 138 patients, accuracy has roughly a ±5 to 6 point margin of error, and results moved by a couple of points between random seeds. A single split is not a stable estimate.
- **Imputation uses the label:** cholesterol zeros were filled with a median computed per diagnosis class on the full dataset, which leaks target information into that feature. `Cholesterol` is not one of the six features in the final model, so the reported results are unaffected, but this should be fixed (see Next steps).
- **Feature selection was done on the full dataset:** it was based on exploratory analysis and correlations across all patients, so the test estimate may be slightly optimistic. Single-feature k-NN scores were unstable between splits (many tied distances on binary features), so I did not use them to rank features.
- **Skewed data:** about 79% of patients are male, so the model may not generalize to women. `Oldpeak` also contains negative values that I did not investigate.
- This is an exercise in applying machine learning, not a clinical tool. Results show associations, not causes.

## Next steps

- Impute missing cholesterol values using training data only (or add a "was missing" flag)
- Compare logistic regression, random forest, and gradient boosting using scikit-learn pipelines and repeated stratified cross-validation
- Report ROC-AUC alongside recall and precision
- Add SQL queries and an interactive dashboard to explore disease rates by age, sex, and chest pain type

## Run it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook heart.ipynb
```

Then use "Restart Kernel and Run All".

## Files

- `heart.ipynb`: cleaning, exploration, modeling, and evaluation
- `heart.csv`: dataset
