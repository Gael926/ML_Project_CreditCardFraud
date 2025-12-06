import pandas as pd
import sys

def load_data(path: str) -> pd.DataFrame:
    # Loads the credit card dataset from a CSV file.
    try:
        print(f"Loading data from {path}...")
        df = pd.read_csv(path)
        print(f"Shape : {df.shape}")
        print(df.info())
        print(f"Total of Null elements : {df.isna().sum().sum()}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        sys.exit(1)

def clean_data(df: pd.DataFrame):
    # Performs initial data cleaning: drops Time column and separates features X and y

    print("Cleaning data")
    
    if 'Time' in df.columns:
        print("Dropping 'Time' column")
        X = df.drop(columns=["Class", 'Time'])
    else:
        X = df.drop(columns=["Class"])
        
    y = df["Class"]
    
    return X, y
