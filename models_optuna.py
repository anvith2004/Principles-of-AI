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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import optuna
from optuna.trial import TrialState
import logging
import optuna.visualization as vis
import traceback
from xgboost import XGBClassifier  # Import XGBoost

# Suppress warnings
warnings.filterwarnings('ignore')

# Prepare directory to store output
output_dir = "files/output"
os.makedirs(output_dir, exist_ok=True)

# Configure logging to log to both console and file
log_file = os.path.join(output_dir, "model_evaluation_optuna.log")
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

# Limit trials per classifier
MAX_TRIALS_PER_CLASSIFIER = 3
classifier_trial_counts = {}

# Define the objective function for Optuna
def objective(trial):
    classifier_name = trial.suggest_categorical('classifier', [
        'random_forest',
        'svc',
        'decision_tree',
        'ada_boost',
        'knn',
        'xgboost',
        'stacking',
        'voting',
        'bagging'
    ])

    # Check if the classifier has reached the max trials
    count = classifier_trial_counts.get(classifier_name, 0)
    if count >= MAX_TRIALS_PER_CLASSIFIER:
        raise optuna.exceptions.TrialPruned()

    # Increment the trial count for the classifier
    classifier_trial_counts[classifier_name] = count + 1

    try:
        # Define the classifier based on trial parameters
        if classifier_name == 'random_forest':
            n_estimators = trial.suggest_int('random_forest_n_estimators', 100, 300, step=100)
            max_depth = trial.suggest_int('random_forest_max_depth', 5, 15)
            min_samples_split = trial.suggest_int('random_forest_min_samples_split', 2, 5)
            classifier_obj = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=1
            )

        elif classifier_name == 'svc':
            C = trial.suggest_float('svc_C', 0.01, 10.0, log=True)
            kernel = trial.suggest_categorical('svc_kernel', ['linear', 'rbf'])
            gamma = trial.suggest_categorical('svc_gamma', ['scale', 'auto'])
            classifier_obj = SVC(
                C=C, kernel=kernel, gamma=gamma, probability=True,
                random_state=42, cache_size=200
            )

        elif classifier_name == 'decision_tree':
            max_depth = trial.suggest_int('decision_tree_max_depth', 3, 10)
            criterion = trial.suggest_categorical('decision_tree_criterion', ['gini', 'entropy'])
            min_samples_split = trial.suggest_int('decision_tree_min_samples_split', 2, 5)
            classifier_obj = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                min_samples_split=min_samples_split,
                random_state=42
            )

        elif classifier_name == 'ada_boost':
            n_estimators = trial.suggest_int('ada_n_estimators', 50, 100, step=50)
            learning_rate = trial.suggest_float('ada_learning_rate', 0.01, 1.0, log=True)
            classifier_obj = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm='SAMME',
                random_state=42
            )

        elif classifier_name == 'knn':
            n_neighbors = trial.suggest_int('knn_n_neighbors', 3, 7)
            weights = trial.suggest_categorical('knn_weights', ['uniform', 'distance'])
            classifier_obj = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                n_jobs=1
            )

        elif classifier_name == 'xgboost':
            n_estimators = trial.suggest_int('xgb_n_estimators', 100, 300, step=100)
            max_depth = trial.suggest_int('xgb_max_depth', 3, 5)
            learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('xgb_subsample', 0.5, 1.0)
            colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
            classifier_obj = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                eval_metric='mlogloss',  # Keep eval_metric to prevent warnings
                random_state=42,
                n_jobs=1
            )

        elif classifier_name == 'stacking':
            # Base estimators
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=50, max_depth=5,
                    random_state=42, n_jobs=1
                )),
                ('svc', SVC(
                    C=1.0, kernel='rbf', probability=True,
                    random_state=42, cache_size=200
                )),
                ('dt', DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                ))
            ]
            # Final estimator
            final_estimator = XGBClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1
            )
            classifier_obj = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                n_jobs=1
            )

        elif classifier_name == 'voting':
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=50, max_depth=5,
                    random_state=42, n_jobs=1
                )),
                ('svc', SVC(
                    C=1.0, kernel='rbf', probability=True,
                    random_state=42, cache_size=200
                )),
                ('knn', KNeighborsClassifier(
                    n_neighbors=5,
                    weights='uniform',
                    n_jobs=1
                ))
            ]
            classifier_obj = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=1
            )

        elif classifier_name == 'bagging':
            n_estimators = trial.suggest_int('bagging_n_estimators', 50, 100, step=50)
            base_estimator = DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
            classifier_obj = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=1
            )

        # Create the pipeline without feature selection
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier_obj)
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, X, y, cv=cv, method='predict', n_jobs=1)

        # Calculate accuracy
        acc = accuracy_score(y, y_pred)
        return acc

    except optuna.exceptions.TrialPruned:
        raise  # Let Optuna handle pruning

    except Exception as e:
        logger.warning(f"Trial failed with error: {e}")
        traceback.print_exc()
        return 0.0  # Penalize the trial with a low score

# Total trials will be MAX_TRIALS_PER_CLASSIFIER * number of classifiers
total_classifiers = 9  # Updated to exclude removed classifiers
total_trials = MAX_TRIALS_PER_CLASSIFIER * total_classifiers

# Create the study with adjusted pruner
study = optuna.create_study(
    direction='maximize',
    study_name='Classifier Optimization',
)

# Run the optimization
study.optimize(objective, n_trials=total_trials, show_progress_bar=True)

# Function to extract best parameters for each classifier
def extract_best_params(study):
    best_params = {}
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            classifier = trial.params.get('classifier')
            if classifier not in best_params:
                best_params[classifier] = {
                    'params': trial.params,
                    'score': trial.value
                }
            else:
                if trial.value > best_params[classifier]['score']:
                    best_params[classifier]['params'] = trial.params
                    best_params[classifier]['score'] = trial.value
    return best_params

# Extract and print the best parameters for each classifier
best_params = extract_best_params(study)
for classifier, info in best_params.items():
    logger.info(f"\nBest parameters for {classifier}:")
    # Filter out the 'classifier' if present
    params = {k: v for k, v in info['params'].items() if k != 'classifier'}
    for param, value in params.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Score: {info['score']:.4f}")

# Initialize DataFrame to store metrics
metrics_results = pd.DataFrame()

# Generate classification reports for best models
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

logger.info("\nClassification Reports for Best Models:\n")

for classifier_name, info in best_params.items():
    # Rebuild the classifier with best parameters
    params = info['params']
    score = info['score']

    # Define the classifier based on best parameters
    if classifier_name == 'random_forest':
        n_estimators = params['random_forest_n_estimators']
        max_depth = params['random_forest_max_depth']
        min_samples_split = params['random_forest_min_samples_split']
        classifier_obj = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=1
        )

    elif classifier_name == 'svc':
        C = params['svc_C']
        kernel = params['svc_kernel']
        gamma = params['svc_gamma']
        classifier_obj = SVC(
            C=C, kernel=kernel, gamma=gamma, probability=True,
            random_state=42, cache_size=200
        )

    elif classifier_name == 'decision_tree':
        max_depth = params['decision_tree_max_depth']
        criterion = params['decision_tree_criterion']
        min_samples_split = params['decision_tree_min_samples_split']
        classifier_obj = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=42
        )

    elif classifier_name == 'ada_boost':
        n_estimators = params['ada_n_estimators']
        learning_rate = params['ada_learning_rate']
        classifier_obj = AdaBoostClassifier(
            n_estimators=n_estimators,
            algorithm='SAMME',
            learning_rate=learning_rate,
            random_state=42
        )

    elif classifier_name == 'knn':
        n_neighbors = params['knn_n_neighbors']
        weights = params['knn_weights']
        classifier_obj = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=1
        )

    elif classifier_name == 'xgboost':
        n_estimators = params['xgb_n_estimators']
        max_depth = params['xgb_max_depth']
        learning_rate = params['xgb_learning_rate']
        subsample = params['xgb_subsample']
        colsample_bytree = params['xgb_colsample_bytree']
        classifier_obj = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=1
        )

    elif classifier_name == 'stacking':
        # Use adjusted estimators
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=50, max_depth=5,
                random_state=42, n_jobs=1
            )),
            ('svc', SVC(
                C=1.0, kernel='rbf', probability=True,
                random_state=42, cache_size=200
            )),
            ('dt', DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ))
        ]
        final_estimator = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1
        )
        classifier_obj = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=1
        )

    elif classifier_name == 'voting':
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=50, max_depth=5,
                random_state=42, n_jobs=1
            )),
            ('svc', SVC(
                C=1.0, kernel='rbf', probability=True,
                random_state=42, cache_size=200
            )),
            ('knn', KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                n_jobs=1
            ))
        ]
        classifier_obj = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=1
        )

    elif classifier_name == 'bagging':
        n_estimators = params['bagging_n_estimators']
        base_estimator = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
        classifier_obj = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=1
        )

    # Create the pipeline without feature selection
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier_obj)
    ])

    # Perform cross-validation to get predictions
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict', n_jobs=1)

    # Check if classifier supports predict_proba
    if hasattr(classifier_obj, "predict_proba"):
        y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=1)
    elif hasattr(classifier_obj, "decision_function"):
        # For classifiers like SVC with probability=False
        decision_scores = cross_val_predict(model, X, y, cv=cv, method='decision_function', n_jobs=1)
        from scipy.special import softmax
        y_proba = softmax(decision_scores, axis=1)
    else:
        # If predict_proba is not available
        y_proba = None

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
        if y_proba is not None:
            auc_macro = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
        else:
            auc_macro = 'Undefined'
    except ValueError:
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
metrics_csv_path = os.path.join(output_dir, "compiled_metrics_optuna.csv")
metrics_results.to_csv(metrics_csv_path, index=False)
logger.info(f"Compiled metrics saved to {metrics_csv_path}")
