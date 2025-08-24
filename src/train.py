from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pickle
import matplotlib.pyplot as plt
import os


def iris_decision_tree():
    
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train,y_train)
    
    path = os.path.join(os.path.dirname(os.getcwd()),'outputs')
    os.makedirs(path,exist_ok=True)
        
    model_path = os.path.join(path, "iris_model.pkl")
    
    ## model saving into pickle file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    
    return model
iris_decision_tree()
