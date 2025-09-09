import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load datasets

# Historical trader data
historical_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2\.venv\historical_data.csv"
historical_df = pd.read_csv(historical_path)
print("Columns in historical data:", historical_df.columns.tolist())

# Parse 'Timestamp IST' with dayfirst=True
historical_df['Time'] = pd.to_datetime(historical_df['Timestamp IST'], dayfirst=True)

# Fear Greed Index data
fear_greed_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2\.venv\fear_greed_index.csv"
fear_greed_df = pd.read_csv(fear_greed_path)
print("Columns in Fear Greed Index:", fear_greed_df.columns.tolist())

# Identify 'Date' column
if 'Date' in fear_greed_df.columns:
    fear_greed_df['Date'] = pd.to_datetime(fear_greed_df['Date'])
elif 'date' in fear_greed_df.columns:
    fear_greed_df['Date'] = pd.to_datetime(fear_greed_df['date'])
else:
    raise ValueError("No 'Date' or 'date' column found in fear_greed_index.csv")

# Identify classification column
classification_column = None
for col in fear_greed_df.columns:
    if col.lower() == 'classification' or 'class' in col.lower() or 'sentiment' in col.lower():
        classification_column = col
        break

if classification_column is None:
    raise ValueError("No classification-related column found in fear_greed_index.csv")

print(f"Using classification column: {classification_column}")

# Step 2: Inspect datasets
print("\nHistorical Data Sample:")
print(historical_df.head())

print("\nFear Greed Index Sample:")
print(fear_greed_df.head())

# Step 3: Preprocess Data

# Extract date from time in historical data to match with fear_greed
historical_df['Date'] = historical_df['Time'].dt.date
fear_greed_df['Date'] = fear_greed_df['Date'].dt.date

# Merge datasets on Date
merged_df = pd.merge(historical_df, fear_greed_df, on='Date', how='left')

print("\nColumns in merged data:", merged_df.columns.tolist())
print("\nMerged Data Sample:")
print(merged_df.head())

# Step 4: Exploratory Analysis

# Check if 'leverage' column exists
leverage_column = None
for col in merged_df.columns:
    if col.lower() == 'leverage':
        leverage_column = col
        break

if leverage_column is None:
    print("Warning: 'leverage' column not found. It will be excluded from aggregation.")
    aggregation_dict = {
        'Closed PnL': 'sum'
    }
else:
    aggregation_dict = {
        'Closed PnL': 'sum',
        leverage_column: 'mean'
    }

# Perform the grouping
summary = merged_df.groupby(['Date', classification_column]).agg(aggregation_dict).reset_index()

print("\nSummary Data:")
print(summary.head())

# Step 5: Visualizations

# Line plot for Closed PnL
plt.figure(figsize=(14,6))
sns.lineplot(data=summary, x='Date', y='Closed PnL', hue=classification_column)
plt.title('Total Trader PnL Over Time by Market Sentiment')
plt.xlabel('Date')
plt.ylabel('Total Closed PnL')
plt.legend(title='Market Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot for Closed PnL
plt.figure(figsize=(10,6))
sns.boxplot(data=merged_df, x=classification_column, y='Closed PnL')
plt.title('Distribution of Closed PnL by Market Sentiment')
plt.xlabel('Market Sentiment')
plt.ylabel('Closed PnL')
plt.tight_layout()
plt.show()

# Boxplot for leverage if it exists
if leverage_column:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=merged_df, x=classification_column, y=leverage_column)
    plt.title('Leverage Usage by Market Sentiment')
    plt.xlabel('Market Sentiment')
    plt.ylabel('Leverage')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping leverage boxplot as the column was not found.")

# Step 6: Correlation Analysis

# Create numeric mapping if applicable
if classification_column:
    # Example mapping – adjust according to your dataset
    mapping = {'Fear': 0, 'Greed': 1}
    if set(mapping.keys()).issubset(merged_df[classification_column].unique()):
        merged_df['Sentiment_Num'] = merged_df[classification_column].map(mapping)
    else:
        print("Warning: Some sentiment labels are not in the mapping. 'Sentiment_Num' may contain NaNs.")

# Prepare list of columns to correlate
corr_columns = ['Closed PnL']
if leverage_column:
    corr_columns.append(leverage_column)
if 'Sentiment_Num' in merged_df.columns:
    corr_columns.append('Sentiment_Num')

correlation = merged_df[corr_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation between PnL, Leverage, and Sentiment')
plt.tight_layout()
plt.show()
