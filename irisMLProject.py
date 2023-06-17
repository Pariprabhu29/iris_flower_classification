# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:02:38 2023

@author: parin
"""

import numpy as np
import pickle as pkl
import streamlit as st
from PIL import Image
filepath="C:/Users/parin/Desktop/irisMLproject/saved_model.sav"
load_model=pkl.load(open(filepath,"br"))
def pred(x):    
    x=np.asarray(x).reshape(1,-1)
    result=load_model.predict(x)    
    if result[0]==0:
        return("SETOSA")
    elif result[0]==1:
        return("VERSICOLOR")
    else:
        return("VIRGINICA")
def main():
    st.markdown('<style>.big-font {font-size:300px ;}</style>', unsafe_allow_html=True)
    st.title(" Iris Flower Classification Project ")
    sl=st.number_input("Sepal-length: ")
    sw=st.number_input("Sepal-width: ")
    pl=st.number_input("petal-length: ")
    pw=st.number_input("petal-width: ")
    data=[sl,sw,pl,pw]
    if st.button("predict"):
        st.write(pred(data))
        if pred(data)=="SETOSA":
            image = Image.open('C:/Users/parin/Desktop/irisMLproject/setosa.jpeg')
            st.image(image, caption='setosa',width=500)
        elif pred(data)=="VERSICOLOR":
            image1 = Image.open('C:/Users/parin/Desktop/irisMLproject/versicolor.jpeg')
            st.image(image1,caption='versicolor',width=500)
        else:
            image2 = Image.open('C:/Users/parin/Desktop/irisMLproject/virginca.jpeg')
            st.image(image2, caption='virginica',width=500)
        
if __name__=="__main__":
    main()    
