## ADDED Requirements

### Requirement: Streamlit-based Spam Classifier UI
The system SHALL provide a Streamlit web UI that allows users to explore the dataset, run realtime inference, and inspect evaluation artifacts produced by a trained model.

#### Scenario: Dataset and column selector
- **WHEN** a user opens the app
- **THEN** the app SHALL allow selection of dataset path and text/label column indices
- **AND** display a preview of the dataset (first 5 rows)

#### Scenario: Class distribution and top terms
- **WHEN** a dataset is selected
- **THEN** the app SHALL display class distribution as a bar chart
- **AND** show top TF-IDF terms per class

#### Scenario: Confusion matrix and ROC/PR
- **WHEN** a trained model exists in `models/` with test split and probabilities saved
- **THEN** the app SHALL display a confusion matrix computed at the selected threshold
- **AND** display ROC and Precision-Recall curves computed from the saved test split

#### Scenario: Threshold slider
- **WHEN** a user adjusts the threshold slider
- **THEN** Precision/Recall/F1 SHALL update to reflect predictions at that threshold

#### Scenario: Realtime inference and quick tests
- **WHEN** a user inputs a message or clicks a quick-test button
- **THEN** the app SHALL perform inference and display predicted label and spam probability
- **AND** quick-test buttons SHALL populate the input without error

### Requirement: Model artifact format
The app SHALL expect `models/` to contain:
- `model.pkl` - a scikit-learn Pipeline supporting `predict_proba` or `decision_function`
- `label_encoder.pkl` - optional LabelEncoder to map integer labels to class names
- `test_split.joblib` - a joblib dict with keys `X_test`, `y_test`, `probs` (probabilities for positive class)
- `metrics.json` - optional summary metrics

#### Scenario: Missing artifacts
- **WHEN** required artifacts are missing
- **THEN** the app SHALL show a clear warning describing how to run `scripts/train_model.py` to produce them

*** End Patch