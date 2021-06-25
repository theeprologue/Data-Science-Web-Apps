# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:49:11 2021

@author: Tholo
"""
import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title('Streamlit example')
st.write("""#Explore Diffrent Classifiers. Which one is best?""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine Dataset'))
st.write(dataset_name)

classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'RandomForest'))

def get_data(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_data(dataset_name)
st.write('Shape of dataset', X.shape)
st.write('Number of Class', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        k = st.sidebar.slider('k', 1, 15)
        params['k'] = k
    elif clf_name == 'SVM':
        c = st.sidebar.slider('k', 0.01, 10.0)
        params['c'] = c
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    elif clf_name == 'SVM':
        clf = SVC(C=params['c'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
		                             max_depth=params['max_depth'])
    return clf

clf = get_classifier(classifier, params)

X_train, X_test, y_train, y_test =train_test_split(X,y)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
st.write(f'classifier: {classifier}')
st.write(f'accuracy: {acc}')

#Plot
pca = PCA(3)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar()
st.pyplot(fig)