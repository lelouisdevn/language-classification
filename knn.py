import pandas as pd

data = pd.read_csv("dataset.csv")

from sklearn.model_selection import train_test_split

import numpy as np

x = np.array(data["Text"])
y = np.array(data["language"])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train,y_train)


text = "Hello world this is text for my machine to identify which language this belongs to Hello world this is text for my machine to identify which language this belongs to"
lang = cv.transform([text]).toarray()
y_pred = model.predict(lang)


print (y_pred)
# print (accuracy_score(y_test, y_pred)*100)
