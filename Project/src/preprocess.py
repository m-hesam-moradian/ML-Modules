from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and apply MinMax scaling
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test
