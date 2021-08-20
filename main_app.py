#importing libraries
import numpy as np 
import streamlit as st 
import cv2
from keras.models import load_model

#loading the model
model = load_model('dog_breed.h5')

#Names of Classes
class_names = ['scottish_deerhound','maltese_dog','afghan_hound']

#Set title of Web app
st.title('Dog Breed Prediction Web App')
st.markdown('Upload an image of Dog')

#Uploading the dog Image
dog_img = st.file_uploader('Choose an image',type='jpg')
submit = st.button('Predict breed')

#on predict button click
if submit:
	if dog_img is not None:

		#convert the file into opencv img
		file_bytes = np.asarray(bytearray(dog_img.read()),dtype=np.uint8)
		opencv_img = cv2.imdecode(file_bytes,1)

		#Displaying the Image
		st.image(opencv_img,channels = 'BGR')
		#Resizing the image
		opencv_img = cv2.resize(opencv_img,(224,224))
		#Convert image into 4 dimentions
		opencv_img.shape = (1,224,224,3)
		#Make Prediction
		Y_pred = model.predict(opencv_img)

		st.title(str('The Dog breed is '+ class_names[np.argmax(Y_pred)]))
