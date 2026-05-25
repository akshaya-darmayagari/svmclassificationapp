import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load artifacts
model = pickle.load(open("model_svc.pkl", "rb"))
scaler = pickle.load(open("scaler_svc.pkl", "rb"))

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

X = df.drop("Species", axis=1)
y = df["Species"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)
pred = model.predict(X_test_scaled)

print("Accuracy Score:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.YlGnBu)
plt.title("SVC: Confusion Matrix")
plt.show()