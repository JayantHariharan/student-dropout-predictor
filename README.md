# Student Dropout & Academic Success Predictor

A Streamlit app that evaluates five classic ML classifiers (Logistic Regression, Decision Tree, KNN,
Naive Bayes, SVM) on a single public dataset, and lets a user upload their own test CSV to see each
model's performance without any retraining happening in the app itself.

## a. Problem statement
Higher-education institutions want to identify students at risk of dropping out early enough to
intervene (tutoring, financial aid, counselling). This project predicts a student's academic
outcome - **Graduate**, **Dropout**, or still **Enrolled** - from demographic, socio-economic and
first/second-semester academic records. This is a **3-class (multi-class)** classification problem.

## b. Dataset description
<!-- DATASET_INFO:START -->
- Source: UCI Machine Learning Repository - Predict Students' Dropout and Academic Success - [https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- Rows: 4424 | Features: 36 | Target: `Target` | Classes: Dropout (32.1%), Enrolled (17.9%), Graduate (49.9%)
- Missing values in features at training time: 0
- Preprocessing: median imputation + standardisation (numeric); most-frequent imputation + one-hot (categorical)
- Split: 80% train / 20% test, stratified, random_state=42 (files in `data/`)
<!-- DATASET_INFO:END -->

## c. Repository and live app
- GitHub repository: <https://github.com/JayantHariharan/student-dropout-predictor>
- Live app (Streamlit Community Cloud): <https://student-dropout-predictor-ml.streamlit.app/>

## d. Models used
<!-- METRICS_TABLE:START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7684 | 0.8961 | 0.7500 | 0.7684 | 0.7531 | 0.6150 |
| Decision Tree | 0.7243 | 0.8534 | 0.7205 | 0.7243 | 0.7216 | 0.5480 |
| KNN | 0.6734 | 0.8154 | 0.6530 | 0.6734 | 0.6576 | 0.4529 |
| Naive Bayes | 0.6588 | 0.8118 | 0.6334 | 0.6588 | 0.6417 | 0.4279 |
| Support Vector Machine | 0.7593 | 0.8681 | 0.7467 | 0.7593 | 0.7458 | 0.6002 |
<!-- METRICS_TABLE:END -->

**How the metrics are computed (3-class problem):** Accuracy and MCC are computed directly on the
predicted labels. Precision, Recall and F1 use *weighted* averaging (per-class scores averaged by class
support), which accounts for the imbalance between Graduate, Dropout and Enrolled. AUC is *one-vs-rest*
(each class against the rest), weighted-averaged, and uses each model's predicted probabilities. All
numbers are measured on the held-out 20% test split (885 rows); no model is retrained on it.

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (Accuracy 0.7684, AUC 0.8961, F1 0.7531, MCC 0.6150). After standardisation the academic features (units approved/enrolled/grade per semester) relate to the outcome close to linearly, so a simple linear model fits well and generalises without overfitting. It catches 92.5% of Graduates and 76.8% of Dropouts, but only 33.3% of Enrolled students. |
| Decision Tree | Third by MCC (0.5480), Accuracy 0.7243. Depth is capped (`max_depth=8`, `min_samples_leaf=5`) to limit overfitting, yet it still trails the linear models. It is the best of the five at the hard Enrolled class (recall 0.421) likely because it can isolate rule-like patterns, but it is less accurate on Dropout (0.683) and Graduate (0.860). Easy to interpret, less stable. |
| KNN | Fourth by MCC (0.4529), Accuracy 0.6734. With 36 features plus one-hot encoded categories, distances become less meaningful (curse of dimensionality), so neighbours are noisy. Recall on Enrolled is only 0.208 because that minority class is outvoted by the larger classes in each neighbourhood (k=7). |
| Naive Bayes | Weakest model: lowest MCC (0.4279), Accuracy 0.6588, F1 0.6417. Gaussian NB assumes features are independent, but many are strongly correlated (e.g. 1st- and 2nd-semester approved units correlate about 0.90 in the data), so probabilities are poorly calibrated. Enrolled recall is just 0.182. It is the fastest model to train. |
| Support Vector Machine | Second by MCC (0.6002), Accuracy 0.7593, close to Logistic Regression (Graduate recall 0.925, Dropout 0.732) but slightly lower on AUC (0.8681 vs 0.8961). The RBF kernel adds flexibility but no real gain here, and its probabilities come from Platt-scaled calibration, which may cost some AUC. Larger and slower than Logistic Regression for no improvement. |
| **Overall winner** | **Logistic Regression** - highest on all six metrics, including MCC (0.6150), which is more robust to class imbalance than accuracy. It is also the simplest, smallest and fastest model, so it wins on performance and practicality. A shared weakness of all five models is the Enrolled class (17.9% of data): best recall is only 0.421, so a future improvement would be class weighting or resampling. |

## Project structure
```
app.py              Streamlit UI - loads saved models, never trains
utils.py             shared validation + metrics helpers
train_models.py      one-time local training script; refreshes the dataset and metrics-table sections of this README
requirements.txt     pinned dependencies (scikit-learn is pinned to the version models were saved with)
data/
  raw.csv            permanent input dataset - never deleted or regenerated
  train_data.csv     80% stratified split - regenerated by train_models.py
  test_data.csv      20% stratified split - regenerated by train_models.py
model/               5 trained pipelines (.joblib) + metadata.json - regenerated by train_models.py
test_data.csv        root copy of the test split - this is the file to upload in the app
metrics_table.csv    numeric metrics table - regenerated by train_models.py
```


## How to run this locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# Train all 5 models (also rewrites the auto-generated sections of this README):
python train_models.py --data data/raw.csv --target Target

# Run the app locally:
streamlit run app.py
```

## Deploying on Streamlit Community Cloud
1. Push this repository to a **public** GitHub repo, including the `model/` folder and `test_data.csv`
   (the app will not work without them - it loads saved models, it does not train).
2. Go to share.streamlit.io -> sign in with GitHub -> New app -> select this repo, branch `main`,
   main file path `app.py` -> Deploy.
3. Once deployed, upload a test CSV (or tick "Use the bundled test_data.csv") to see metrics, a
   confusion matrix, a classification report, and per-model predictions.

## Using this app
1. Upload a CSV containing all feature columns plus the target column (or use the bundled sample).
2. Pick a model from the sidebar dropdown.
3. View its metrics, confusion matrix and classification report in the first tab, compare all 5
   models in the second tab, and inspect/download row-level predictions in the third tab.

## Contributing
This repository is public for assignment-review purposes. If you spot a bug or want to extend it
(e.g. add another dataset or model), the usual flow applies: open an issue describing the change,
fork the repo, make your change on a branch, and open a pull request against `main`. Please do
not commit retrained model files without also updating `metrics_table.csv` and re-running
`train_models.py` so the README stays in sync with the committed `model/` folder.

## Reproducibility notes (for re-running or re-grading)
- All randomness is controlled by `random_state=42` in `train_models.py`; re-running training on the
  same `data/raw.csv` reproduces the exact metrics above.
- `app.py` and `utils.py` contain no `.fit(` calls - the Streamlit app only loads and predicts with the
  models already committed under `model/`.
