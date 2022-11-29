import pytesseract


def imageToText(img):
    text = pytesseract.image_to_string(img)
    return text


print (imageToText("./handwritting.jpg"))
