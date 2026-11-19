import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/Clean_Dataset.csv")

# Check if dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError("Dataset not found!")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("INT395 - AIRFARE COMPONENT PRICE ESTIMATION")
print("DATASET INSPECTION")
print("=" * 60)

# Dataset size
print("\n1. DATASET DIMENSIONS")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column names
print("\n2. COLUMN NAMES")
print(df.columns.tolist())

# First five rows
print("\n3. FIRST FIVE RECORDS")
print(df.head())

# Data types
print("\n4. DATA TYPES")
print(df.dtypes)

# Missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# Duplicate records
print("\n6. DUPLICATE RECORDS")
print("Duplicates:", df.duplicated().sum())

# Statistical summary
print("\n7. STATISTICAL SUMMARY")
print(df.describe())

# Price analysis
print("\n8. PRICE ANALYSIS")
print("Minimum price:", df["price"].min())
print("Maximum price:", df["price"].max())
print("Average price:", df["price"].mean())
print("Median price:", df["price"].median())

print("\nDATASET INSPECTION COMPLETED!")
