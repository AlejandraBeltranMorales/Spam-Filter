import zipfile
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

z = zipfile.ZipFile("spam2-train.zip")

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

z_test = zipfile.ZipFile("spam2-test.zip")
emails_test = []
file_names = []

for name in z_test.namelist():
    content = str(z_test.read(name))
    emails_test.append(content)
    file_names.append(name) 

# Step 5: Extract features for unknown test emails using the same vectorizer
features_test = cv.transform(emails_test)

# Step 6: Use the model to classify unknown emails
predictions = model.predict(features_test)

with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')

    for name, prediction in zip(file_names, predictions):
        classification = "1" if prediction == 1 else "0"
        # print(f"{name};{classification}")
        writer.writerow([name, prediction])

print("Archive .csv exported successfuly")
