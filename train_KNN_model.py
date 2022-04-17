'''
K Nearest Neighbours Classifier model for recognizing Handwritten digits
written on 4/12/2022 by kshitijv256

'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import dump, load, pick

# Load dataset
training_data = load("./datasetX3.csv")

ytrain = training_data["784"]
xtrain = training_data.drop(labels=["784"],axis = 1)

del training_data

xtrain = xtrain/255.0
x_train,x_val,y_train,y_val = train_test_split(xtrain,ytrain,test_size=0.1)
print()

# KNN model

print('Training the model...\n')
model = KNeighborsClassifier(n_neighbors=3,algorithm='auto',n_jobs=20)
model.fit(x_train,y_train)

print('Saving the model\n')

dump(model,'./saved/KNN.pickle')
model = pick('./saved/KNN.pickle')

# Testing on previously splitted data

print("Testing on data\n")
results = model.predict(x_val)
accuracy = accuracy_score(y_val,results)

print('Accuracy is: ',accuracy)