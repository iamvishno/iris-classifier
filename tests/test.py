from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt 
import os

#dataset
dataset = load_iris()
X = dataset.data
y = dataset.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


path = os.path.join(os.path.dirname(os.getcwd()),'outputs')
model_path = os.path.join(path, "iris_model.pkl")

with open(model_path,'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Accuracy', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))          
sns.heatmap(cm, annot=True, fmt='d',cmap='viridis',xticklabels=dataset.target_names,yticklabels=dataset.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

path = os.path.join(os.path.dirname(os.getcwd()),'outputs')
os.makedirs(path,exist_ok=True)

confusion_metrix_path = os.path.join(path,'confusion_metrix.jpg')

plt.savefig(confusion_metrix_path)