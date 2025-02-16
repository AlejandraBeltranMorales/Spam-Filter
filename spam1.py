import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import csv

z = zipfile.ZipFile("spam1-train.zip")

emails = []

labels = []

for name in z.namelist():
    if name.endswith("labels"):
        continue
    content = str(z.read(name))
    emails.append(content)
    label = int(name[-1])
    labels.append(label)

emails_train, labels_train = emails[0:8000], labels[0:8000]
emails_test, labels_test = emails[8000:], labels[8000:]

# they need numbers to work --> Extracting features from emails
cv = CountVectorizer()
features = cv.fit_transform(emails_train)

# Build a model
tuned_parameters = {'kernel':['linear'],'gamma':[1e-3],'C':[1]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features, labels_train) 
#train the model --> when you do .fit is already trained

z_test = zipfile.ZipFile("spam1-test.zip")
emails_test = []
file_names = []

for name in z_test.namelist():
    content = str(z_test.read(name))
    emails_test.append(content)
    file_names.append(name) 

features_test = cv.transform(emails_test)

predictions = model.predict(features_test)

with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')

    for name, prediction in zip(file_names, predictions):
        classification = "1" if prediction == 1 else "0"
        # print(f"{name};{classification}")
        writer.writerow([name, prediction])

print("Archive .csv exported successfuly")
