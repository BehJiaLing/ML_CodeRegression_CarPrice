# =============================================
# Step 1: Import libraries
# =============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================
# Step 2: Load dataset
# =============================================
df = pd.read_csv("datasets/train-data.csv")

# Drop unnamed index column if it exists
if df.columns[0].startswith("Unnamed"):
    df = df.drop(df.columns[0], axis=1)
    print("Removed unnamed first column")

print("Original shape:", df.shape)
print(df.head())

# =============================================
# Step 3: Data cleaning - remove text units and handle numeric conversion
# =============================================

# Helper function to extract numeric values safely
def extract_number(x):
    if pd.isnull(x):
        return np.nan
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

# Clean unit-based columns
if 'Mileage' in df.columns:
    df['Mileage'] = df['Mileage'].apply(lambda x: str(x).replace('km/kg', '').replace('kmpl', '').strip())
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

if 'Engine' in df.columns:
    df['Engine'] = df['Engine'].apply(lambda x: str(x).replace('CC', '').strip())
    df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')

if 'Power' in df.columns:
    df['Power'] = df['Power'].apply(lambda x: str(x).replace('bhp', '').strip())
    df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

if 'New_Price' in df.columns:
    df['New_Price'] = df['New_Price'].apply(lambda x: str(x).replace('Lakh', '').strip())
    df['New_Price'] = pd.to_numeric(df['New_Price'], errors='coerce')

print("\nSample cleaned numeric columns:")
print(df[['Mileage', 'Engine', 'Power']].head())

# =============================================
# Step 4: Handle missing values
# =============================================
print("\nMissing values before handling:")
print(df.isnull().sum())

# Drop target-missing rows
df = df.dropna(subset=['Price'])

# Drop 'New_Price' since 86% missing
if 'New_Price' in df.columns:
    missing_ratio = df['New_Price'].isnull().mean()
    if missing_ratio > 0.7:
        df = df.drop('New_Price', axis=1)
        print("Dropped 'New_Price' (86% missing values)")

# Fill remaining missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after handling:")
print(df.isnull().sum())

# =============================================
# Step 5: Remove duplicates
# =============================================
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"\nRemoved {before - after} duplicate rows")

# =============================================
# Step 6: Feature Engineering
# =============================================
# Extract brand from Name
if 'Name' in df.columns:
    df['Brand'] = df['Name'].apply(lambda x: str(x).split()[0])
    df = df.drop('Name', axis=1)

# =============================================
# Step 7: Handle categorical and numerical columns
# =============================================
cat_cols = ['Fuel_Type', 'Owner_Type', 'Transmission', 'Location', 'Brand']
num_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# Merge encoded columns
df = pd.concat([df.drop(cat_cols, axis=1).reset_index(drop=True),
                encoded_df.reset_index(drop=True)], axis=1)

print("\nEncoded categorical columns added:")
print(encoded_df.head())

# =============================================
# Step 8: Handle outliers
# =============================================
for col in ['Year', 'Kilometers_Driven', 'Price']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = df.shape[0]
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"Removed outliers in {col}: {before - df.shape[0]} rows")

# =============================================
# Step 9: Scale numerical features
# =============================================
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nScaled numerical columns:")
print(df[num_cols].head())

# =============================================
# Step 10: Log-transform target (reduce skewness)
# =============================================
df['Price'] = np.log1p(df['Price'])

# =============================================
# Step 11: Export cleaned dataset
# =============================================
df.to_csv("datasets/train-processed-data.csv", index=False)
print("\n✅ Preprocessing complete. Saved as 'train-processed-data.csv'")
print("Final shape:", df.shape)
