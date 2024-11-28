import pandas as pd
from sklearn.impute import KNNImputer
from scipy.stats.mstats import winsorize

pd.set_option('display.max_colwidth', None)  # To display full column width
pd.set_option('display.max_columns', None)   # To display all columns
pd.set_option('display.width', 1000)

# Load the dataset
df = pd.read_csv(r'files/Final.csv', encoding='ISO-8859-1')

# Define the list of features to extract and create a copy to avoid SettingWithCopyWarning
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

# Convert all columns except 'Name', 'Sector', and 'Sub-Sector' to numeric, forcing errors to NaN
for col in features.columns:
    if col not in ['Name', 'Sector', 'Sub-Sector']:
        features[col] = pd.to_numeric(features[col], errors='coerce')

print(features.head())

# Number of duplicated rows
duplicated_rows_count = features.duplicated().sum()
print("Number of duplicated rows:", duplicated_rows_count)
print("\n\n")

# Number of NaN values per column
nan_counts_initial = features.isna().sum()
print("Number of NaN values per column before imputation:\n", nan_counts_initial)
print('\n\n')

# Impute with mean for continuous columns
continuous_columns = [
    'Return on Equity', 'ROCE', 'Net Profit Margin', 'Return on Assets',
    'Asset Turnover Ratio', 'Current Ratio', 'PE Ratio', 'PB Ratio',
    'Debt to Equity', 'Debt to Asset', 'PS Ratio'
]
for col in continuous_columns:
    features[col] = features[col].fillna(features[col].mean())

# Impute the remaining specified columns to 0
zero_fill_cols = [
    'Return on Investment', 'MF Holding Change 3M', 'MF Holding Change 6M',
    'FII Holding Change 3M', 'FII Holding Change 6M', 'DII Holding Change 3M',
    'DII Holding Change 6M', 'EPS (Q)', 'Dividend Per Share'
]
for col in zero_fill_cols:
    features[col] = features[col].fillna(0)

# Use KNN Imputer for specific columns with complex missing patterns
imputer = KNNImputer(n_neighbors=5)
features[['Market Cap', 'EBITDA Margin', 'Working Capital Turnover Ratio']] = imputer.fit_transform(
    features[['Market Cap', 'EBITDA Margin', 'Working Capital Turnover Ratio']]
)

# Winsorization to handle outliers
def winsorize_columns(df, columns, lower_percentile=0.05, upper_percentile=0.95):
    """
    Apply Winsorization to specified columns in the DataFrame.
    Caps values at the specified lower and upper percentiles.
    """
    for col in columns:
        df[col] = winsorize(df[col], limits=(lower_percentile, 1 - upper_percentile))
    return df

# Apply Winsorization to continuous columns
features = winsorize_columns(features, continuous_columns, lower_percentile=0.05, upper_percentile=0.95)

# Optionally, sort the counts by class label
class_counts = features['R2'].value_counts()
class_counts = class_counts.sort_index()

# Display the counts
print(class_counts)
print()

# Check for remaining missing values after imputation and Winsorization
nan_counts_final = features.isna().sum()
print("Remaining Missing Values per Column after Winsorization:\n", nan_counts_final)

# Display the shape of the dataset after preprocessing
print("Shape of the dataset after preprocessing:", features.shape)

# Save the preprocessed DataFrame to a new CSV file
features.to_csv('files/Final_Preprocess1_winsorize.csv', index=False)
print("Preprocessed data saved as 'Final_Preprocess1_winsorize.csv'")
print()
