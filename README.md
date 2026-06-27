# Model Validation, Overfitting Control & Hyperparameter Tuning

**Task 3 — AI & Machine Learning**
Detecting overfitting, validating models reliably with cross-validation, and tuning a Decision Tree with `GridSearchCV` on the California Housing dataset.

---

## 📁 Dataset

[California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) — 20,640 rows, 9 columns, no missing values.

| Features | Target |
|---|---|
| `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude` | `MedHouseVal` (median house value) |

## 🛠️ Tools

Python · pandas · NumPy · scikit-learn · matplotlib

## 🔁 Workflow

1. Load & explore the dataset, check for missing values
2. Scale features with `StandardScaler`
3. 80/20 train-test split
4. Train an **unconstrained Decision Tree** to deliberately expose overfitting
5. Run **5-fold cross-validation** for a reliable performance estimate
6. Tune the tree with **`GridSearchCV`** over `max_depth` and `min_samples_split`
7. Compare the tuned tree against **Linear Regression** and **Ridge Regression**
8. Select and justify the final model
9. Bonus: actual-vs-predicted plot and residual analysis

## 📊 Key Results

**Overfitting check (untuned Decision Tree):**

| | Train RMSE | Test RMSE |
|---|---|---|
| Untuned Tree | 0.0000 | 0.7060 |

A train RMSE of ~0 against a test RMSE of 0.706 is the classic overfitting signature — the unconstrained tree memorized the training data. 5-fold CV confirmed this generalizes poorly (mean CV RMSE **0.9029**, std 0.0326).

**GridSearchCV tuning:**

Grid searched: `max_depth ∈ {3, 5, 7, 10}`, `min_samples_split ∈ {2, 5, 10}`

→ Best params: **`max_depth=10`, `min_samples_split=10`** (CV RMSE **0.6366**)

**Final model comparison:**

| Model | Train RMSE | Test RMSE | CV RMSE (5-fold) | R² |
|---|---|---|---|---|
| Linear Regression | 0.7197 | 0.7456 | 0.7459 | 0.5758 |
| Ridge Regression | 0.7197 | 0.7456 | 0.7459 | 0.5758 |
| **Tuned Decision Tree** | 0.0000* | **0.6454** | 0.7948 | **0.6821** |

\* Train RMSE shown is from the unconstrained baseline tree in the overfitting check — once `max_depth=10` is applied, train error is no longer near-zero, since the tree can no longer memorize every point.

## ✅ Final Model: Tuned Decision Tree

- **Lowest test RMSE** (0.6454) and **highest R²** (0.6821) of all three models
- Overfitting controlled by capping `max_depth` at 10
- Improvement confirmed by cross-validation, not just a single lucky split
- Hyperparameters chosen systematically via `GridSearchCV`, not by hand
- Ridge regularization matched plain Linear Regression exactly, showing the linear models' ceiling is structural — they can't capture non-linear pricing patterns — not a variance problem regularization could fix

**Caveat:** CV RMSE (0.7948) is noticeably higher than the single-split test RMSE (0.6454), and residuals show more spread at higher predicted values, so real-world performance should be read as a range rather than a fixed number.

## 📂 Repo Contents

```
├── AI_ML_Task3_Model_Validation_Tuning.ipynb   # Full notebook
├── california_housing.csv                       # Dataset
└── README.md
```

## ▶️ Running the Notebook

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook AI_ML_Task3_Model_Validation_Tuning.ipynb
```
