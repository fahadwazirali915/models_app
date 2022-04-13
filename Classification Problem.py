# import libraries !
import streamlit as st
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import pandas as pd
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Heading 
st.info('''
         ## Explore differnet ML-models and Datasets !
         #### **Developed by Fahad !**
         ''')


# Sidebar
# > selectbox for datasets
datasets_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Breast Cancer', 'Iris', 'Wine Quality')
)
# > selectbox for models
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'Random Forest', 'SVM')
)


# Load Datasets
def get_dataset(datasets_name):
    df = None
    if datasets_name == 'Breast Cancer':
        df = datasets.load_breast_cancer()
    elif datasets_name == 'Iris':
        df = datasets.load_iris()
    else:
        df = datasets.load_wine()
    # split dataset !
    X = df.data
    y = df.target
    return X, y    
# get dataset. (method calling)
X, y = get_dataset(datasets_name)


# print shape & uniqueness of (y) of dataset
st.write("Shape of dataset: ", X.shape)
st.write("No's of classes: ", len(np.unique(y)))


# Parameters of different classifiers
def add_parameter_ui(classifier_name):
    params = dict()    # empty dictionary
    if classifier_name == 'KNN':
        K = st.sidebar.slider('Neighbours', 1, 15)
        params['K'] = K    # its the number of nearest neighbours
    elif classifier_name == 'SVM':
        C = st.sidebar.slider('Degree', 1, 15)
        params['C'] = C    # its the degree of correct classification
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth    # depth of every tree that grow in Random-Forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators    # number of trees    
    return params
# get sliders/parameters. (method calling)
params = add_parameter_ui(classifier_name)
             
    
# get classifiers
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(max_depth=params['max_depth'],
                                     n_estimators=params['n_estimators'],
                                     random_state=1234)
    return clf
# get classifiers. (method calling)
clf = get_classifier(classifier_name, params)


# splitting dataset into train test
X_train,X_test , y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# train classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# check accuracy
acc = np.round(accuracy_score(y_test, y_pred)*100, 4)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = {acc}')

st.markdown("***")

# plot dataset
# using PCA.(feature reduction technique)==[all feature into 2-dimensional plot.(scatter plot)]
pca = PCA(2)
X_projected = pca.fit_transform(X) 

# dataset slice into (0 & 1) dimension
x1 = X_projected[:, 0] 
x2 = X_projected[:, 1]

# plt.figure(figsize=(4,4)) 
fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')    # (c act like hue), (cmap define 'color'), (alpha shows transparency)
plt.xlabel('Principal Component 1')   
plt.ylabel('Principal Component 2')
plt.colorbar()

st.write("### Scatter-plot")
st.pyplot(fig)    # show figure   
