# Early Sepsis Diagnosis with Machine Learning

<p align="center">
  <img src="https://github.com/user-attachments/assets/a312d1bb-32e9-4ffa-b625-a00f202c5fb4" alt="image" />
</p>

## Overview

This project presents a machine learning framework for early sepsis diagnosis, developed using state-of-the-art algorithms and evaluated against key clinical adoption criteria: predictive power, explainability, and economic value. The framework is based on data from the PhysioNet 2019 Challenge, a widely recognized ML competition featuring real ICU data from multiple hospital systems.

The implementation includes a comparative analysis of temporal neural networks (LSTM, TCN) and gradient-boosted tree models (CatBoost, XGBoost, LightGBM). Model performance is benchmarked against top-performing teams from the PhysioNet Challenge and leading open-source implementations, such as [nerajbobra’s solution](https://github.com/nerajbobra/sepsis-prediction-physionet), demonstrating competitive or superior results in key metrics, including AUC-ROC.

Explainable AI libraries—SHAP and DALEX—are employed to assess feature contributions and validate alignment with medical literature. An economic simulation is included to estimate the potential cost savings of early AI-supported diagnosis. The project contributes a ready-to-adapt framework for real world deployment of AI-based decision support systems in healthcare.


---

## Contents

- `Sepsis_ML_Article.pdf`  
  *Full thesis, based on which a research article is to appear in* [*Journal of Health Economics*](https://www.sciencedirect.com/journal/journal-of-health-economics)

- `scripts/`
  - `Custom_Training_Algorithm_Test.ipynb`  
    *Notebook validating algorithm optimizing moving window selection for training*
  - `dataloader_wrapper_cuda.py`  
    *Custom PyTorch dataloader implementing the method from the training algorithm notebook*
  - `Sepsis_Prediction_GBTs_Eval.ipynb`  
    *Gradient Boosted Trees (GBT) training and evaluation*
  - `Sepsis_Prediction_NNs.ipynb`
  - `Sepsis_Prediction_NNs_Logs.ipynb` <br/>
    *Neural Network (NN) training and evaluation (version with and w/o outputs)*
  - `Recall_In_Hours_To_Sepsis.ipynb`  
    *Analysis notebook for recall as a function of time to sepsis onset*


---

## Objectives

- Investigate predictive performance of multiple ML models for early sepsis detection
- Validate model decisions against established medical knowledge using XAI tools
- Quantify the economic advantage of earlier AI-based diagnosis
- Propose a framework aligning with AI-DDS (AI-based Diagnostic Decision Support) adoption requirements

---

## Dataset

- **Source**: [PhysioNet 2019 Challenge](https://physionet.org/content/challenge-2019/1.0.0/)
- **Features**: 41 vital signs, lab values, and demographic indicators, recorded hourly
- **Size**: ~813,000 observations across 40,000+ ICU patients
- **Target**: Sepsis onset (binary classification)

---

## Methods

### Modeling Pipeline

- **Preprocessing**: Bayesian Ridge imputation, interaction terms, first differences, scaling
- **Models Evaluated**:
  - Neural Nets: ANN, LSTM, TCN
  - Gradient Boosted Trees: CatBoost, XGBoost, LightGBM
- **Training Setup**:
  - LSTM and TCN: Moving windows, binary cross-entropy loss, weighted for class imbalance
  - GBTs: Grid search optimization with 5-fold cross-validation

### Explainability

- **Tools**: SHAP (Shapley values), DALEX (drop-out loss, PDP)
- **Focus**: Feature contribution analysis (e.g., ICULOS, heart rate, DBP), trust alignment with medical literature

### Economic Evaluation

- Cost-saving simulations using US hospitalization data (Paoli et al., Liang et al.)
- Threshold optimization for maximizing economic benefit
- Use of `hour_effect` multiplier to estimate relative savings over time
- Acknowledgement of the fact that sepsis is "an absorbing state" in evaluations

---

## Results

### Predictive Power

| Model     | AUC-ROC (Val) | AUC-ROC (Test) |
|-----------|---------------|----------------|
| LSTM      | 0.853         | 0.859          |
| CatBoost  | 0.860         | 0.857          |
| XGBoost   | 0.850         | —              |
| TCN       | 0.835         | —              |
| ANN       | 0.827         | —              |

- **LSTM** and **CatBoost** delivered the best predictive scores.
- **CatBoost** was selected for final deployment due to training speed and higher economic efficiency.
- Recall as a function of time was additionally calculated for the best-performing models in the notebook `Recall_In_Hours_To_Sepsis.ipynb`.

### Explainability

- ICULOS, heart rate, and gender were dominant risk predictors.
- Features aligned well with known clinical indicators (e.g., DBP, hematocrit, EtCO₂).
- SHAP and PDP visualizations helped confirm model logic.

### Economic Advantage

- **Estimated Savings (Validation Set)**:
  - CatBoost: Up to \$6.97M
  - LSTM: Up to \$6.70M
- Cost savings increased with earlier detection (modeled via `hour_effect` parameter).

---

## Key Findings

- Gradient-boosted trees (CatBoost) achieved a strong balance of accuracy, explainability, and cost-effectiveness slightly outperforming NN models.
- LSTM outperformed TCN, challenging the assumption that convolutional architectures offer better temporal modeling in this context.
- Explainability analysis confirmed medical plausibility, helping bridge the gap between model output and clinical intuition.

---

## Business & Clinical Impact

- Early detection of sepsis may reduce ICU costs by up to \$2.8M annually (based on case studies).
- Trust-building through XAI techniques enhances clinical acceptance of AI diagnostics.
- Validated pipeline supports integration into EHR-based early warning systems.

---

## Limitations

- Significant missing data were handled via imputation, but still pose a problem
- Simplified economic modeling (linear cost proxy via `hour_effect`)
