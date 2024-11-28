import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    BaggingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
import logging
from xgboost import XGBClassifier
from sklearn.exceptions import ConvergenceWarning
from scipy.special import softmax

# Suppress warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Prepare directory to store output
output_dir = "files/output"
os.makedirs(output_dir, exist_ok=True)

# Configure logging to log to both console and file
log_file = os.path.join(output_dir, "model_evaluation_no_optuna.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create formatter and add to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Load the dataset
df = pd.read_csv(r'files/Final_Preprocess1_winsorize.csv', encoding='ISO-8859-1')

# Define the list of features to extract
features = df[[
    'Name', 'Sector', 'Sub-Sector', 'Market Cap', 'Return on Equity', 'ROCE',
    'Cash Flow Margin', 'EBITDA Margin', 'Net Profit Margin', 'Return on Assets',
    'Asset Turnover Ratio', 'Working Capital Turnover Ratio', 'Current Ratio',
    'Debt to Equity', 'PE Ratio', 'PB Ratio', 'Sector PE', 'Sector PB', 'PS Ratio',
    'Sector Dividend Yield', 'Return on Investment', 'MF Holding Change 3M',
    'MF Holding Change 6M', 'FII Holding Change 3M', 'FII Holding Change 6M',
    'DII Holding Change 3M', 'DII Holding Change 6M', 'EPS (Q)', 'Dividend Per Share',
    'Debt to Asset', 'R2'
]].copy()

# Convert all columns except 'Name', 'Sector', and 'Sub-Sector' to numeric
for col in features.columns:
    if col not in ['Name', 'Sector', 'Sub-Sector']:
        features[col] = pd.to_numeric(features[col], errors='coerce')

# Drop rows with missing target variable
features.dropna(subset=['R2'], inplace=True)

# Define target variable and features
X = features.drop(columns=['Name', 'R2', 'Sub-Sector'])
y = features['R2']

# Identify numerical and categorical features
categorical_features = ['Sector']
numerical_features = X.columns.difference(categorical_features)

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Initialize DataFrame to store metrics
metrics_results = pd.DataFrame()

# List of classifiers with default or suboptimal parameters
classifiers = {
    'random_forest': RandomForestClassifier(
        n_estimators=10, max_depth=2, min_samples_split=10, class_weight=None, random_state=42, n_jobs=1
    ),
    'svc': SVC(
        C=1e-3, kernel='poly', gamma='auto', probability=True, class_weight=None, random_state=42
    ),
    'decision_tree': DecisionTreeClassifier(
        max_depth=2, criterion='entropy', min_samples_split=10, class_weight=None, random_state=42
    ),
    'ada_boost': AdaBoostClassifier(
        n_estimators=10, learning_rate=1.0, algorithm='SAMME.R', random_state=42
    ),
    'knn': KNeighborsClassifier(
        n_neighbors=15, weights='uniform', n_jobs=1
    ),
    'xgboost': XGBClassifier(
        n_estimators=10, max_depth=2, learning_rate=1.0, subsample=0.5, colsample_bytree=0.5,
        eval_metric='mlogloss', random_state=42, n_jobs=1
    ),
    'stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=10, max_depth=2, random_state=42, n_jobs=1
            )),
            ('svc', SVC(
                C=1e-3, kernel='poly', gamma='auto', probability=True, random_state=42
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=2, criterion='entropy', min_samples_split=10, random_state=42
            ))
        ],
        final_estimator=RandomForestClassifier(
            n_estimators=10, max_depth=2, random_state=42, n_jobs=1
        ),
        n_jobs=1
    ),
    'voting': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=10, max_depth=2, random_state=42, n_jobs=1
            )),
            ('svc', SVC(
                C=1e-3, kernel='poly', gamma='auto', probability=True, random_state=42
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=2, criterion='entropy', min_samples_split=10, random_state=42
            ))
        ],
        voting='soft',
        n_jobs=1
    ),
    'bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2, random_state=42
        ),
        n_estimators=10,
        random_state=42,
        n_jobs=1
    )
}

# Cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

logger.info("\nClassification Reports for Models without Optuna Optimization:\n")

for classifier_name, classifier_obj in classifiers.items():
    # Create the pipeline without LDA
    pipeline_steps = [
        ('preprocessor', preprocessor),
        ('classifier', classifier_obj)
    ]
    model = Pipeline(steps=pipeline_steps)

    # Perform cross-validation to get predictions
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict', n_jobs=1)
    y_proba = None
    if hasattr(classifier_obj, "predict_proba"):
        y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=1)
    elif hasattr(classifier_obj, "decision_function"):
        decision_scores = cross_val_predict(model, X, y, cv=cv, method='decision_function', n_jobs=1)
        # Convert decision scores to probabilities using softmax
        y_proba = softmax(decision_scores, axis=1)

    # Generate classification report
    report = classification_report(y, y_pred, digits=4)
    logger.info(f"Classification Report for {classifier_name}:\n{report}")

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)

    # Calculate AUC-ROC macro average
    try:
        auc_macro = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
    except (ValueError, TypeError):
        auc_macro = 'Undefined'

    # Calculate AUC-ROC for each class
    n_classes = len(np.unique(y))
    auc_scores = {}
    for i in range(n_classes):
        y_true_binary = (y == i).astype(int)
        if y_proba is not None and y_proba.shape[1] > i:
            y_score = y_proba[:, i]
            try:
                auc_score = roc_auc_score(y_true_binary, y_score)
                auc_scores[f'Class {i}'] = auc_score
            except ValueError:
                auc_scores[f'Class {i}'] = 'Undefined (Only one class present in y_true)'
        else:
            auc_scores[f'Class {i}'] = 'Undefined'

    logger.info(f"AUC-ROC Scores for {classifier_name}:")
    for cls, auc_score in auc_scores.items():
        logger.info(f"  {cls}: {auc_score}")
    logger.info(f"Macro AUC-ROC: {auc_macro}\n")

    # Compile all metrics
    metrics = {
        'Model': classifier_name,
        'Accuracy': accuracy,
        'Precision (Macro)': precision,
        'Recall (Macro)': recall,
        'F1 Score (Macro)': f1,
        'AUC-ROC (Macro)': auc_macro
    }
    # Add per-class AUC-ROC scores to metrics
    for cls, auc_score in auc_scores.items():
        metrics[f'AUC-ROC {cls}'] = auc_score

    # Append metrics to results DataFrame
    metrics_results = pd.concat([metrics_results, pd.DataFrame([metrics])], ignore_index=True)

# Save the compiled metrics to CSV
metrics_csv_path = os.path.join(output_dir, "compiled_metrics_no_optuna.csv")
metrics_results.to_csv(metrics_csv_path, index=False)
logger.info(f"Compiled metrics saved to {metrics_csv_path}")
