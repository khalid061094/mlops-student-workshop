from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
def train_model():
iris = load_iris()
model = LogisticRegression(
max_iter=300
)
model.fit(
iris.data,
iris.target
)
return model
if __name__ == "__main__":
model = train_model()
print("Model trained successfully!")
