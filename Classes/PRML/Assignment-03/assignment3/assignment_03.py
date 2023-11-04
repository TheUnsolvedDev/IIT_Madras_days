import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from os import walk
from nltk.corpus import stopwords
# from striprtf.striprtf import rtf_to_text

# Training on a general online dataset
data = pd.read_csv('spam.csv')
data.info()

X = data['EmailText'].values
y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Converting String to Integer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Linear Kernel
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

print("Linear Kernel:")
print(classifier.score(X_test,y_test))

# Polynomial Kernel
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

print("Polynomial Kernel:")
print(classifier.score(X_test,y_test))

# RBF Kernel
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

print("RBF Kernel:")
print(classifier.score(X_test,y_test))

# Sigmoid Kernel
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

print("Sigmoid Kernel:")
print(classifier.score(X_test,y_test))

# Training on Custom Dataset from Kaggle
pathwalk = walk("enron-data/")
SpamData,HamData = [],[]

for root,dr,files in pathwalk:
    if "spam" in str(files):
        for file in files:
            with open(root + '/' + file,encoding='latin1') as ip:
                SpamData.append(" ".join(ip.readlines()))
    if "ham" in str(files):
        for file in files:
            with open(root + '/' + file,encoding='latin1') as ip:
                HamData.append(" ".join(ip.readlines()))

SpamData = list(set(SpamData))
HamData = list(set(HamData))
Data = SpamData + HamData
Labels = ["spam"]*len(SpamData) + ["ham"]*len(HamData)

raw_df = pd.DataFrame({
    "email":Data,
    "label":Labels
})

stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stopWords)

email = vectorizer.fit_transform(raw_df.email.to_list())
label_encoder = sk.preprocessing.LabelEncoder()
labels = label_encoder.fit_transform(raw_df.label)

X_train,X_test,y_train,y_test = train_test_split(email,labels,train_size=0.8,random_state=42,shuffle=True)

# Linear Kernel
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

print("Linear Kernel:")
print(classifier.score(X_test,y_test))

# Polynomial Kernel
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

print("Polynomial Kernel:")
print(classifier.score(X_test,y_test))

# RBF Kernel
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

print("RBF Kernel:")
print(classifier.score(X_test,y_test))

# Sigmoid Kernel
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

print("Sigmoid Kernel:")
print(classifier.score(X_test,y_test))

# # Test Folder Reading Function
# def read_test_emails():
#     testwalk = walk("test/")
#     emails = []
#     for root,dr,files in testwalk:
# #         print(f"{root},{dr},{files}")
#         for file in files:
#             if "email"  in file:
#     #             print(file)
#                 with open(root + file) as infile:
#                     content = infile.read()
#                     text = rtf_to_text(content)
#                 emails.append(text)
#     return emails
#
# test_vectorizer = CountVectorizer(stop_words=stopWords)#, min_df=1)
#
# X_test_ = test_vectorizer.fit_transform(read_test_emails()).toarray()
# X_test_mod = np.zeros((X_test_.shape[0],X_train.shape[1]))
#
# # print(vectorizer.vocabulary_)
#
# dict_train = vectorizer.vocabulary_
# dict_test = test_vectorizer.vocabulary_
#
# for key in dict_train.keys():
#     if key in dict_test.keys():
# #         print(f"train:{dict_train[key]},test:{dict_test[key]}")
#         X_test_mod[:,dict_train[key]] = X_test_[:,dict_test[key]]X_test_mod
#
# print(y_pred)
