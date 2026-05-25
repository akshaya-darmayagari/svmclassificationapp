import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

# Split features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
pickle.dump(model, open("model_svc.pkl", "wb"))
pickle.dump(scaler, open("scaler_svc.pkl", "wb"))

print("SVC Model Saved Successfully")