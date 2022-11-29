from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tkinter import *
from imagePr import *
from tkinter import filedialog
from PIL import ImageTk, Image


# read dataset from file
data = pd.read_csv("dataset.csv")
print (data[0:5])

# build model
x = np.array(data["Text"])
y = np.array(data["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, y_train)


# application - GUI
# methods:
def browseFile():
    # return 0
    filename = filedialog.askopenfilename(initialdir="/home/louis/Documents/CT46604_NLCNTT/lab/", title="Select a File", filetypes=(("Image","*.png*"),("all files","*.*")))
    return filename


# 1. process: du doan nhan cho phan tu
def process(data):
    data = cv.transform([data]).toarray()
    result = model.predict(data)
    return result


#convert image to text
def identify():
    # get image from user.
    img = browseFile()
    image1 = Image.open(img)
    image1 = image1.resize((500, 200), Image.ANTIALIAS)
    test = ImageTk.PhotoImage(image1)

    # display the image
    photolabel = Label(image=test)
    photolabel.image = test
    photolabel.grid (row=1, column=0)

    # extracts text from image
    data = imageToText(img)
    # get result from identification process
    result = process(data)

    if data == "\n":
        result = "No input data!"
    else:
        str = ""
        for e in result:
            str = e
        result = "Detected: " + str

    # display the language detected
    output.set(result)
    l2.grid(row=0, column=1)

    tb1 = Text(height=12, width=40)
    tb1.insert(END, data)
    tb1.grid(row=1, column=1)


# build app:
app = Tk()
app.title("Natural Language Identification")
app.geometry("850x400")

l1 = Label(app, text="Select an image:", font=("Arial", 15))
l1.grid(row=0, column=0)

output = StringVar()
l2 = Label(app, textvariable=output, font=("arial", 15))

btn = Button(app, text="Open", command=identify)
btn.grid(row=17, column=0)

close = Button(app, text="Close", command=app.destroy, activebackground="red")
close.grid(row=17, column=1)

app.mainloop()
