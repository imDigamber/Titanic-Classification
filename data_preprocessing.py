# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load the dataset
    titanic_data = pd.read_csv(file_path)

    # Handling missing values
    titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)

    # Encoding categorical variables
    titanic_data['sex'] = titanic_data['sex'].map({'male': 0, 'female': 1})
    titanic_data = pd.get_dummies(titanic_data, columns=['embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], drop_first=True)

    # Feature selection
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked_Q', 'embarked_S', 'class_Second', 'class_Third', 'who_man', 'adult_male_True', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'embark_town_Queenstown', 'embark_town_Southampton', 'alive_yes', 'alone_True']
    X = titanic_data[features]
    y = titanic_data['survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
