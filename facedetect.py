import numpy as np #NumPy is a python library used for working with arrays. It also has functions for working in domain of linear algebra,
# fourier transform, and matrices. ... NumPy stands for Numerical Python.
import cv2 #Open Source Computer Vision Library , designed to solve computer vision problems and image processing
#OpenCV-Python makes use of Numpy, which is a highly optimized library for numerical operations with a MATLAB-style syntax.
# All the OpenCV array structures are converted to and from Numpy arrays.
import pickle #Pickle in Python is primarily used in serializing and deserializing a Python object structure.

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#machine learning object detection algorithm used to identify objects in an image or video
recognizer = cv2.face.LBPHFaceRecognizer_create()#The Local Binary Pattern Histogram(LBPH) algorithm is a simple solution on face recognition problem
recognizer.read("trainning.yml") #yml" is "the file extension for the YAML file format,in which the data of technology is stored
labels = {"person_name":1}
with open("labels.pickle",'rb') as f: # opening the label.pickle file
    og_labels = pickle.load(f)  #file loaded
    labels = {v:k for k,v in og_labels.items()}#serializzing the names


cap = cv2.VideoCapture(0)#0 for webcam

while 1: # when webcam open
    ret, img = cap.read() # start reading from the webcame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color image into gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #face detection width and height

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) #rectangle color,stroke
        roi_gray = gray[y:y + h, x:x + w] #its means rectangle start on face and end , on x and y axix in gray format
        roi_color = img[y:y + h, x:x + w]  #its means rectangle start on face and end , on x and y axix in color format

        id_,conf=recognizer.predict(roi_gray) # prediction
        if conf>=1: #and conf <=85:  #confidence
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX # font
            name = labels[id_] #
            color = (255,255,255) # name color
            stroke = 2 # name stroke
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA) # putting the name of person in the right top of rectangle
            print("done")
        else:
            print("error")

        img_item = "my-image.png"
        cv2.imwrite(img_item,roi_color) # saving the image which is currently recognition




    cv2.imshow('img', img)     # webcame window
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()