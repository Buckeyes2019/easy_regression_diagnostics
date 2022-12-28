"""Main module."""

from sklearn.linear_model import LinearRegression

def run_regression(x, y):
    """
    Runs a linear regression.py

    Arguments:
        - x(array): numeric features from a numpy array
        - y(vector): numeric target
    
    """
    model = LinearRegression().fit(x,y)
    
    return model.score(x,y)

