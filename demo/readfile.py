import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

#read file with pandas
data = pd.read_csv("./dataset.csv")

print ("The first 5 piece of data:")
print (data.head())
print ("____________________________"
        "______________________________")

#extract attributes and label
x = np.array(data["Text"])
y = np.array(data["language"])

print ("x: ", x)
print ("y: ", y)

print ("____________________________"
        "______________________________")

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
# X = cv.fit_transform(x[0:10])
# print (X.toarray())
X = cv.fit_transform(x)


# print ("____________________________"
        # "______________________________")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

print ("data: ", )
print ("X_train:", X_train.shape)
print ("X_test: ", X_test.shape)
print ("y_train", y_train.shape)
print ("y_test:", y_test.shape)


print ("_________________________")

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print ("Accuracy score: ", accuracy_score(y_test, y_pred)*100, "%")
# textdata = "Ce système peut utiliser " + \
#             " un texte comme entrée pour spécifier à quelle langue il appartient"
textdata = input("Enter some text:...")
lang = cv.transform([textdata]).toarray()
result = model.predict(lang)
print (result)