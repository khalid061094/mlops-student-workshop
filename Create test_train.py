# Import training function
from train import train_model


# Define test function
def test_model_can_predict():

    # Train the model
    model = train_model()

    # Make a prediction
    prediction = model.predict(
        [[5.1, 3.5, 1.4, 0.2]]
    )

    # Check prediction output
    assert len(prediction) == 1

    print("Test passed! Model can make predictions.")


# Run test
test_model_can_predict()
