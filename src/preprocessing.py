from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from src.config import RANDOM_STATE

def get_train_test_split(X, y):
    # Splits the data into training and testing sets with stratification.
    return train_test_split(X, y, stratify=y, random_state=RANDOM_STATE)

def get_preprocessor():
    print("Creating preprocessing pipeline")
    
    features_to_scale = ["Amount"]
    preprocess = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), features_to_scale)
        ],
        remainder='passthrough'
    )
    
    return preprocess
