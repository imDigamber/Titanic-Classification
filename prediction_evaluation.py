# prediction_evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import preprocess_data
from model_training import train_model

if __name__ == "__main__":
    # Use an absolute path for gender_submission.csv
    file_path = r'D:\Titanic Classification\gender_submission.csv'

    # Data preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Model training
    trained_model = train_model(X_train, y_train)

    # Prediction and evaluation
    y_pred = trained_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", class_report)
