# model_training.py

from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    # Build and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model
