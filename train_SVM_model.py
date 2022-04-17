'''
K Nearest Neighbours Classifier model for recognizing Handwritten digits
written on 4/12/2022 by kshitijv256

'''
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import dump, load, pick

# Load data
training_data = load("./dataset.csv")

ytrain = training_data['784']
xtrain = training_data.drop(labels=['784'],axis = 1)

del training_data

xtrain = xtrain/255.0
x_train,x_val,y_train,y_val = train_test_split(xtrain,ytrain,test_size=0.1)
print()

# SVM model

print('Training the model...\n')
model = svm.SVC(gamma=0.1, kernel='poly')
model.fit(x_train,y_train)

print('Saving the model\n')

dump(model,'./saved/SVM2.pickle')
model = pick('./saved/SVM2.pickle')

# Testing on previously splitted data

print("Testing on data\n")
results = model.predict(x_val)
accuracy = accuracy_score(y_val,results)

print('Accuracy is: ',accuracy)