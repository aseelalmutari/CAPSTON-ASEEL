# Vantrex: Context-Aware IoT Cyber Activity Detection
## Project Report

**Author:** Aseel Almutairi  
**Dataset:** TON-IoT Network Dataset  
**Task:** Binary Classification — Normal vs. Attack  
**Primary Metric:** F1-Score

---

## 1. Introduction

IoT devices are deployed across homes, hospitals, industrial systems, and smart cities. Unlike traditional computers, most IoT devices lack built-in security software, making them attractive targets for attackers. Common attacks against IoT networks include backdoors, distributed denial-of-service (DDoS), scanning, password attacks, injection, ransomware, and man-in-the-middle attacks.

**Vantrex** is a machine learning system designed to detect abnormal behavior in IoT network traffic. It analyzes network traffic features, device telemetry, and contextual metadata to classify each network event as either **Normal** or **Attack**. The system is trained on the TON-IoT Network Dataset and deployed as a local Streamlit web application where users can enter feature values or upload a CSV file to receive predictions with confidence scores and risk levels.

**Why this matters:** In IoT environments, millions of events are generated per day. Manual review is impossible, and missed attacks can lead to data breaches, ransomware, and physical damage to infrastructure. Automated detection enables rapid response.

**Primary metric:** F1-score, because the dataset is class-imbalanced. A model that naively predicts "Attack" for every record would achieve ~76% accuracy without learning anything — F1-score prevents this by requiring both high Precision and high Recall.

---

## 2. Dataset Description

**Dataset:** TON-IoT Network Dataset  
**File:** `data/train_test_network.csv`

| Property | Value |
|---|---|
| Total Rows | 211,043 |
| Columns | 44 |
| Duplicate Rows | ~20,569 (removed before training) |
| Rows After Deduplication | ~190,474 |
| Target Column | `label` (0 = Normal, 1 = Attack) |
| Normal Records | ~50,000 (~24%) |
| Attack Records | ~161,000 (~76%) |

**Class Imbalance:** Attack records outnumber Normal records by approximately 3:1. This means accuracy alone is a misleading metric — F1-score was used as the primary evaluation metric because it balances Precision and Recall.

**Dataset Challenges:**

1. **No true NaN values** — missing data appears as the string `'-'` in categorical columns (protocol fields irrelevant to the current connection type, e.g., DNS fields for a plain TCP connection). Several numeric columns (dns_qclass, dns_qtype, dns_rcode, http_trans_depth, etc.) also contained `'-'` and required conversion with `pd.to_numeric(..., errors='coerce')`.

2. **Duplicate rows** — 20,569 exact duplicate records were removed before the train/test split to prevent identical rows from appearing in both training and evaluation sets.

3. **Leakage columns** — the following columns were excluded from features:
   - `src_ip`, `dst_ip` — would allow the model to memorize specific hosts rather than learn behavioral patterns
   - `type` — directly encodes attack category (backdoor, ddos, etc.), which would trivially inflate model performance
   - `dns_query`, `ssl_subject`, `ssl_issuer`, `http_uri`, `http_user_agent`, `weird_addl` — high-cardinality free-text fields, mostly `'-'`

4. **Mixed data types** — 17 numeric features and 17 categorical features required separate preprocessing pipelines combined using `ColumnTransformer`. In total, 34 features were used for training.

**Preprocessing approach:**
- Numeric columns: `SimpleImputer(strategy='median')` → `StandardScaler()`
- Categorical columns: `SimpleImputer(strategy='constant', fill_value='missing')` → `OneHotEncoder(handle_unknown='ignore')`
- The full `ColumnTransformer` was fitted **only on training data** to prevent data leakage. Test data was transformed using training statistics only.

---

## 3. Models Used

Three scikit-learn classifiers were trained and evaluated. All used `class_weight='balanced'` to compensate for the 3:1 class imbalance, and all were wrapped in a `Pipeline` with the same `ColumnTransformer` preprocessor.

### Logistic Regression

**Role:** Baseline linear classifier  
**Configuration:** `max_iter=1000, class_weight='balanced', solver='lbfgs'`  
**Why chosen:** Fast, interpretable, produces probability outputs, well-suited for binary classification. Serves as a lower-bound reference for more complex models.

The `class_weight='balanced'` setting adjusts loss weights to compensate for class imbalance, preventing the model from ignoring the minority (Normal) class.

### Decision Tree

**Role:** Interpretable rule-based model  
**Configuration:** `max_depth=15, class_weight='balanced', random_state=42`  
**Why chosen:** Provides transparent decision logic that can be explained in plain language. Demonstrates how features split the two classes. `max_depth=15` limits overfitting — unlimited depth would produce near-perfect training scores but poor generalization.

### Random Forest

**Role:** Ensemble model — expected strongest classifier  
**Configuration:** `n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1`  
**Why chosen:** Robust against overfitting through ensemble averaging of 100 decision trees. Handles nonlinear relationships and feature interactions, which are common in cybersecurity network data. Strong performance on tabular datasets is well-established.

---

## 4. Neural Network

A Keras Sequential Dense Neural Network was built as the deep learning component.

**Architecture:**
```
Input(shape=(N_features,))
Dense(128, activation='relu')
Dropout(0.3)
Dense(64, activation='relu')
Dropout(0.3)
Dense(32, activation='relu')
Dense(1, activation='sigmoid')
```

**Design choices:**
- **ReLU** hidden layers: prevents vanishing gradients, computationally efficient, standard for dense tabular networks
- **Sigmoid** output: produces a probability in [0, 1] for the binary Attack/Normal decision
- **Dropout(0.3)**: randomly deactivates 30% of neurons during training to reduce overfitting
- **Adam optimizer** (learning_rate=0.001): adaptive learning rate, fast convergence on tabular data
- **Binary Crossentropy** loss: appropriate for binary classification tasks

**Training:**
- Up to 20 epochs, batch size 256
- 20% of training data held as validation split
- `EarlyStopping(patience=5)` restores best weights when validation loss stops improving
- Ensures minimum 10 epochs run before early stopping triggers

**Improvement Experiment:** Dropout rate was increased from 0.3 to 0.4 in a second training run (V2). Both models were evaluated on validation loss and the lower-loss model was selected for test evaluation. This demonstrates the impact of regularization strength on generalization.

Training and validation loss/accuracy curves were plotted (saved in `figures/nn_training_curves.png`) to detect overfitting and verify learning. The curves showed consistent convergence with no significant gap between training and validation metrics, indicating the Dropout layers successfully controlled overfitting.

**Deployment:** The trained NN is saved as `models/nn_model.keras` and can be toggled on in the local Streamlit web application (`app/app.py`) alongside the sklearn pipeline for side-by-side comparison.

### Lightweight NLP Assistant

A lightweight rule-based NLP assistant was added to the local Streamlit application to help users understand the project outputs in a simple way. The assistant accepts short text questions from the user, applies basic text preprocessing such as lowercasing, punctuation removal, and whitespace normalization, then matches the cleaned question to predefined intents using keyword-based rules.

The chatbot supports simple project-related questions about Normal and Attack predictions, F1-score, Random Forest model selection, confidence score, risk level, the Neural Network model, the TON-IoT dataset, and how to use the application. This feature is inspired by the Week 9 NLP concepts, especially text preprocessing, tokenization, intent detection, and simple chatbot behavior.

The assistant is intentionally rule-based and does not use an external API, LLM, or separately trained NLP model. Its purpose is to improve user understanding and make the Streamlit interface more educational, while the main prediction task remains handled by the trained Machine Learning and Neural Network models.

---

## 5. Results and Comparison

All four models were evaluated on the same held-out 20% test set using the same preprocessing pipeline fitted only on training data.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.9985** | **0.9987** | **0.9993** | **0.9990** | **0.9999** |
| Decision Tree | 0.9970 | 0.9977 | 0.9985 | 0.9981 | 0.9937 |
| Neural Network (Keras) | 0.9944 | 0.9954 | 0.9975 | 0.9964 | 0.9993 |
| Logistic Regression | 0.9861 | 0.9943 | 0.9879 | 0.9911 | 0.9951 |

All models were trained on 80% of the deduplicated dataset (~152,379 rows) and evaluated on the held-out 20% test set (~38,095 rows). Confusion matrices for all four models are saved in `figures/`.

**Primary metric: F1-Score**

F1-Score was chosen as the primary metric because:
- The dataset is imbalanced (~76% Attack)
- A model achieving 76% accuracy by always predicting "Attack" would have 0% Recall for Normal records and is useless in practice
- F1-Score requires both Precision (few false alarms) and Recall (few missed attacks) to be high simultaneously

**Security metric priority for intrusion detection:**
1. **Recall** — most important: a missed attack (False Negative) is far more dangerous than a false alarm (False Positive)
2. **Precision** — too many false alarms reduce trust in the system and cause alert fatigue
3. **F1-Score** — harmonic mean that balances both

**Best model: Random Forest** — selected and saved as `models/best_pipeline.pkl` because:
1. Highest F1-Score (0.9990) and highest Recall (0.9993) across all models
2. Near-perfect ROC-AUC (0.9999), indicating excellent separation between Normal and Attack classes
3. Saved as a single sklearn `Pipeline` (preprocessor + classifier bundled together), eliminating preprocessing mismatch risk at deployment
4. Faster inference than the Neural Network — no TensorFlow dependency at runtime

The Neural Network achieved strong results (F1 = 0.9964, ROC-AUC = 0.9993) and is saved separately as `models/nn_model.keras`. The Streamlit app provides a sidebar toggle to switch between the sklearn pipeline and the Keras NN for comparison.

---

## 6. What I Learned

**1. Data cleaning is the foundation of every ML project.**  
The TON-IoT dataset appeared clean (0 NaN values) but contained ~20,569 duplicate rows and thousands of `'-'` placeholder strings. Without removing duplicates before the train/test split, evaluation metrics would be inflated. Without correctly handling `'-'` strings, encoding would fail or produce wrong categories.

**2. F1-score is the right metric for imbalanced datasets.**  
Using accuracy on a 76%/24% imbalanced dataset is misleading. A 90%+ accuracy sounds great but can hide a model that completely fails on the minority class. For intrusion detection, the cost of missing an attack is far greater than the cost of a false alarm — Recall on the Attack class matters most.

**3. Preprocessing must match exactly between training and deployment.**  
The biggest practical risk in ML deployment is a mismatch between how data was preprocessed during training and how it is preprocessed at inference time. Saving a full sklearn `Pipeline` (which bundles `ColumnTransformer` + classifier together) eliminates this risk entirely — the app calls `pipeline.predict_proba(raw_df)` on raw input and gets the same result as training, because the pipeline applies identical preprocessing automatically.

**4. More complex models are not always better.**  
The Neural Network adds significant training time and hyperparameter complexity. If Random Forest achieves comparable or higher F1-Score, the sklearn model is preferable for deployment: faster inference, no TensorFlow dependency, simpler debugging. Complexity should be justified by measurable improvement.

**5. Data leakage is subtle and must be deliberately prevented.**  
The `type` column (attack category) and IP address columns could trivially inflate model performance during training while providing no real detection capability in deployment. Identifying and removing these leakage sources required domain understanding — not just looking at the data structure.

**6. Class imbalance requires intentional handling.**  
Using `class_weight='balanced'` in all sklearn models automatically adjusts for the 3:1 Attack/Normal ratio. Without this, the models would be biased toward always predicting "Attack" (the majority class), achieving high raw accuracy but poor real-world utility.

**Future work:**  
Multi-class attack type classification using the `type` column (currently excluded to prevent leakage — could be a separate supervised learning task), cloud deployment via Streamlit Community Cloud, and real-time packet capture integration.
