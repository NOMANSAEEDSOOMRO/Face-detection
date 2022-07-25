import os,cv2
#The OS module in python provides functions for interacting with the operating system.
#Open Source Computer Vision Library , designed to solve computer vision problems and image processing
#OpenCV-Python makes use of Numpy, which is a highly optimized library for numerical operations with a MATLAB-style syntax.
# All the OpenCV array structures are converted to and from Numpy arrays.
from PIL import Image #Python Imaging Library (abbreviated as PIL) load image , import , open ,rename, filename
import numpy as np #NumPy is a python library used for working with arrays. It also has functions for working in domain of linear algebra,
# fourier transform, and matrices. ... NumPy stands for Numerical Python.
import pickle #Pickle in Python is primarily used in serializing and deserializing a Python object structure.

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #current location of project
image_dir = os.path.join(BASE_DIR,"images") # current location joining with the image folder

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')##machine learning object detection algorithm used to identify objects in an image or video
recognizer = cv2.face.LBPHFaceRecognizer_create()##The Local Binary Pattern Histogram(LBPH) algorithm is a simple solution on face recognition problem

current_id = 0 #initializing currentid with 0
labels_ids = {} #dictionary object of labelids
y_label = [] # label list
x_train = [] # train list
for root,dirs,files in os.walk(image_dir):
    for file in files: #its means images inside the folder noman saeed soomro
        if file.endswith("jpg") or file.endswith("png"): #file type jpg or png
            path=os.path.join(root,file)  #joining the folder of image with the subfolder
            label = os.path.basename(root).replace("","") #replacing the name of images with ""
            #print(label,path)
            if not label in labels_ids:
              labels_ids[label]= current_id
              current_id +=1 # increment if labels
            id_ = labels_ids[label]
            #print(labels_ids)
            #y_label.append(label)
            #x_train.append(path)
            pil_image=Image.open(path).convert("L")#grayscale
            image_array = np.array(pil_image,"uint8")# converting image into numpy array
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5) #face detection
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h,x:x+w] #image array into the rectangle
                x_train.append(roi) #passing the roi obj into existing list
                y_label.append(id_) #passing the id_ obj into existing list

#print(y_label)
#print(x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(labels_ids,f) # for storing the pickle file

recognizer.train(x_train,np.array(y_label))
recognizer.save("trainning.yml") # for storing the training file






        


