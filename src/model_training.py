from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_train_test(X_train, X_test, exclude_cols=None):
    """
    fit scaler for train data，and transform train + test
    exclude_cols
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if exclude_cols is None:
        exclude_cols = []

    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    scaling_cols = [col for col in numerical_cols if col not in exclude_cols]

    scaler = StandardScaler()
    scaler.fit(X_train[scaling_cols])
    X_train_scaled[scaling_cols] = scaler.transform(X_train[scaling_cols])
    X_test_scaled[scaling_cols] = scaler.transform(X_test[scaling_cols])

    return X_train_scaled, X_test_scaled