from django.shortcuts import render, render_to_response
from tkinter import filedialog
from tkinter import *
from django.core.files.storage import FileSystemStorage


# Script Packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import pandas as pd
import urllib
import cv2

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19, preprocess_input


# Create your views here.


def index(request):

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'index.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'index.html')


# Algorithm Runs here
         # result = 10.5
         # return render(request, "index.html", {'result': result})


# Extra Code


def upload_local(request, localimage):
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    print(root.filename)
    return
    # render_to_response("index.html", RequestContext(request, {}))

# upload_local()


# Model Selection
def Models(request):
    return render(request, 'Models.html')


# ResNet50 Model
def ResNet50_Model(request):
    # model_ResNet50 = ResNet50(weights='imagenet')
    return render(request, 'ResNet50_Model.html')

# VGG19 Model


def VGG19_Model(request):
    model_VGG19 = VGG19(weights='imagenet')
    return render(request, 'VGG19_Model.html')

# ResNet50_Local


def ResNet50_Local(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'ResNet50_Local.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:

            return render(request, 'ResNet50_Local.html')
    return render(request, 'ResNet50_Local.html')

# ResNet50_URL


def ResNet50_URL(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'ResNet50_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, '<ResNet50_URL></ResNet50_URL>.html')
    return render(request, 'ResNet50_URL.html')

    # url = input('Enter a JPG or PNG URL: ')
    # url = "https://image.freepik.com/free-photo/hrc-siberian-tiger-2-jpg_21253111.jpg"
    # url_response = urllib.request.urlopen(url)  # extract the contents of the URL
    # image_url = image.load_img(url_response, target_size=(224, 224))  # img = image.load_img(img, target_size=(224, 224))
    # return render(request, 'ResNet50_URL.html')

    """
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'ResNet50_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'ResNet50_URL.html')
    return render(request, 'ResNet50_URL.html')
    """

# VGG19_Local


def VGG19_Local(request):
    if request.method == 'POST' and request.FILES['myfile2']:
        myfile2 = request.FILES['myfile2']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile2.name, myfile2)
        uploaded_file_url = fs.url(filename)
        return render(request, 'VGG19_Local.html', {'uploaded_file_url': "static/" + myfile2.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'VGG19_Local.html')
    return render(request, 'VGG19_Local.html')


# VGG19_URL
def VGG19_URL(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'VGG19_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:

            return render(request, 'VGG19_URL.html')
    return render(request, 'VGG19_URL.html')

#________________________________________________________________________________________________________________
# ResNet50_Local_Predict


def ResNet50_Local_Predict(request):
    # model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('fileurl')
    # print(request.POST)
    if request.method == 'POST':
        my_param = "C:\\Users\\Kevin D'Cruz\\Downloads\\UMass\\GitHub\\Image Recognition System API\\IRS_API\\IRS_API\\" + request.POST.get('url')
        img = image.load_img(my_param, target_size=(224, 224))

        plot_prediction(img)

        # print(my_param)

        if my_param is None:
            return render(request, 'ResNet50_Local_Predict.html')
    return render(request, 'ResNet50_Local_Predict.html')

    """
    k = request.GET.get('uploaded_file_url')
    print(k)
    return render(request, 'ResNet50_Local_Predict.html')


    plot_prediction()
    return render(request, 'ResNet50_Local_Predict.html')
    """
# Prediction function: ResNet50 local image


def predict(model_ResNet50, img, target_size, top_n=5):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_ResNet50.predict(x)
    return decode_predictions(preds, top=top_n)[0]

# Plotting: resNet50 Local Image


def plot_prediction(img):
    model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    #k = request.POST.get('url')
    prediction = predict(model_ResNet50, img, target_size=(224, 224))  # Model prediction
    prediction = pd.DataFrame(np.array(prediction).reshape(5, 3), columns=list("abc"))  # Converting to a DataFrame
    print("This picture has the highest possibility of a " + '\033[1m' '\033[4m' + prediction.b[0])
    graph = prediction.convert_objects(convert_numeric=True)
    display(graph)


"""
# plt.bar(graph.b, graph.c, align='center', color='gray', edgecolor='black', width=0.4)
# plt.rcParams['figure.figsize'] = 13, 6
# plt.xlabel("Predicted Outcomes", color='blue')
# plt.ylabel("Output Probabilities", color='blue')
# plt.show()
#________________________________________________________________________________________________________________________
"""
