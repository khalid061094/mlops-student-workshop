# Import required libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# Define training function
def train_model():

    # Load Iris dataset
    iris = load_iris()

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=300
    )

    # Train the model
    model.fit(
        iris.data,
        iris.target
    )

    return model


# Main program
if __name__ == "__main__":

    model = train_model()

    print("Model trained successfully!")
