import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(df):
    numerical_features = ['Age', 'Education-num',
                          'Capital-gain', 'Capital-loss', 'Hours-per-week']
    categorical_features = ['Workclass', 'Education', 'Marital-status', 'Occupation',
                            'Relationship', 'Race', 'Sex', 'Native-country']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False,
             handle_unknown='ignore'), categorical_features)
        ])

    X_transformed = preprocessor.fit_transform(df)
    return X_transformed, preprocessor.get_feature_names_out()


if __name__ == "__main__":
    print("Start Preprocessing")
    column_names = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num',
                    'Marital-status', 'Occupation', 'Relationship', 'Race',
                    'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week',
                    'Native-country', 'Income']
    df = pd.read_csv("Data/adult.csv", names=column_names)

    X, names = preprocess_data(df)
    print("Preproccessing done")
