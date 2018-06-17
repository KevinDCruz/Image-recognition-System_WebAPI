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
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions


# Create your views here.

# ----------------------------------Index View-------------------------------------------------------------------
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
#_____________________________________________________________________________________________________

# Algorithm Runs here
         # result = 10.5
         # return render(request, "index.html", {'result': result})


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


#------------------------------------ ResNet50 Model------------------------------------------------------------------------
def ResNet50_Model(request):
    # model_ResNet50 = ResNet50(weights='imagenet')
    return render(request, 'ResNet50_Model.html')


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

# Plotting: ResNet50 Local Image


def plot_prediction(img):
    model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('url')
    prediction = predict(model_ResNet50, img, target_size=(224, 224))  # Model prediction
    prediction = pd.DataFrame(np.array(prediction).reshape(5, 3), columns=list("abc"))  # Converting to a DataFrame
    graph = prediction.convert_objects(convert_numeric=True)
    print(graph)
    print("This picture has the highest possibility of a " + '\033[1m' '\033[4m' + prediction.b[0])
    # display(graph)
    plt.bar(graph.b, graph.c, align='center', color='gray', edgecolor='black', width=0.4)
    plt.rcParams['figure.figsize'] = 13, 6
    plt.xlabel("Predicted Outcomes", color='blue')
    plt.ylabel("Output Probabilities(* 100 for percent)", color='blue')
    plt.savefig('IRS_API/static/ResNet50_Local.jpg', bbox_inches='tight')
    # plt.savefig('myapp/templates/ResNet50_Local.jpg', bbox_inches='tight')

#________________________________________________________________________________________________________________________

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


#--------------------------------------- VGG19 Model------------------------------------------------------------------


def VGG19_Model(request):
    # model_VGG19 = VGG19(weights='imagenet')
    return render(request, 'VGG19_Model.html')


# VGG19_Local


def VGG19_Local(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'VGG19_Local.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'VGG19_Local.html')
    return render(request, 'VGG19_Local.html')


# VGG19_Local_Predict


def VGG19_Local_Predict(request):
    # model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('fileurl')
    # print(request.POST)
    if request.method == 'POST':
        my_param = "C:\\Users\\Kevin D'Cruz\\Downloads\\UMass\\GitHub\\Image Recognition System API\\IRS_API\\IRS_API\\" + request.POST.get('url')
        img = image.load_img(my_param, target_size=(224, 224))

        plot_prediction_VGG19_Local(img)

        # print(my_param)

        if my_param is None:
            return render(request, 'VGG19_Local_Predict.html')
    return render(request, 'VGG19_Local_Predict.html')


def predict_VGG19(model_VGG19, img, target_size, top_n=5):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_VGG19.predict(x)
    return decode_predictions(preds, top=top_n)[0]

# Plotting: ResNet50 Local Image


def plot_prediction_VGG19_Local(img):
    model_VGG19 = VGG19(weights='imagenet')  # Model Load
    # k = request.POST.get('url')
    prediction = predict_VGG19(model_VGG19, img, target_size=(224, 224))  # Model prediction
    prediction = pd.DataFrame(np.array(prediction).reshape(5, 3), columns=list("abc"))  # Converting to a DataFrame
    graph = prediction.convert_objects(convert_numeric=True)
    # print(graph)
    statement = print("This picture has the highest possibility of a " + '\033[1m' '\033[4m' + prediction.b[0])
    # print(statement)
    # display(graph)
    plt.bar(graph.b, graph.c, align='center', color='gray', edgecolor='black', width=0.4)
    plt.rcParams['figure.figsize'] = 13, 6
    plt.xlabel("Predicted Outcomes", color='blue')
    plt.ylabel("Output Probabilities(* 100 for percent)", color='blue')
    plt.savefig('IRS_API/static/VGG19_Local.jpg', bbox_inches='tight')
    # plt.savefig('myapp/templates/ResNet50_Local.jpg', bbox_inches='tight')s

#____________________________________________________________________________________________________________________________________

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

    #----------------VGG16 Model------------------------------------------------------------------


def VGG16_Model(request):
    # model_VGG19 = VGG19(weights='imagenet')
    return render(request, 'VGG16_Model.html')


# VGG16_Local
def VGG16_Local(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'VGG16_Local.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'VGG16_Local.html')
    return render(request, 'VGG16_Local.html')


# VGG16_Local_Predict


def VGG16_Local_Predict(request):
    # model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('fileurl')
    # print(request.POST)
    if request.method == 'POST':
        my_param = "C:\\Users\\Kevin D'Cruz\\Downloads\\UMass\\GitHub\\Image Recognition System API\\IRS_API\\IRS_API\\" + request.POST.get('url')
        img = image.load_img(my_param, target_size=(224, 224))

        plot_prediction_VGG16_Local(img)

        # print(my_param)

        if my_param is None:
            return render(request, 'VGG16_Local_Predict.html')
    return render(request, 'VGG16_Local_Predict.html')


def predict_VGG16(model_VGG16, img, target_size, top_n=5):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_VGG16.predict(x)
    return decode_predictions(preds, top=top_n)[0]

# Plotting: ResNet50 Local Image


def plot_prediction_VGG16_Local(img):
    model_VGG16 = VGG16(weights='imagenet')  # Model Load
    # k = request.POST.get('url')
    prediction = predict_VGG16(model_VGG16, img, target_size=(224, 224))  # Model prediction
    prediction = pd.DataFrame(np.array(prediction).reshape(5, 3), columns=list("abc"))  # Converting to a DataFrame
    graph = prediction.convert_objects(convert_numeric=True)
    graph.to_csv('VGG16_Local_Graph.csv')

    print(graph)
    statement = print("This picture has the highest possibility of a " + '\033[1m' '\033[4m' + prediction.b[0])
    # print(statement)
    # display(graph)
    plt.bar(graph.b, graph.c, align='center', color='gray', edgecolor='black', width=0.4)
    plt.rcParams['figure.figsize'] = 13, 6
    plt.xlabel("Predicted Outcomes", color='blue')
    plt.ylabel("Output Probabilities(* 100 for percent)", color='blue')
    plt.savefig('IRS_API/static/VGG16_Local.jpg', bbox_inches='tight')
    # plt.savefig('myapp/templates/ResNet50_Local.jpg', bbox_inches='tight')s


#____________________________________________________________________________________________________________________________________

# VGG19_URL


def VGG16_URL(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'VGG16_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:

            return render(request, 'VGG16_URL.html')
    return render(request, 'VGG16_URL.html')

#____________________________________________________________________________________________________________________________________

   #----------------InceptionV3 Model------------------------------------------------------------------


def InceptionV3_Model(request):
    # model_VGG19 = VGG19(weights='imagenet')
    return render(request, 'InceptionV3_Model.html')


# InceptionV3_Local
def InceptionV3_Local(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'InceptionV3_Local.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'InceptionV3_Local.html')
    return render(request, 'InceptionV3_Local.html')


# InceptionV3_Local_Predict


def InceptionV3_Local_Predict(request):
    # model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('fileurl')
    # print(request.POST)
    if request.method == 'POST':
        my_param = "C:\\Users\\Kevin D'Cruz\\Downloads\\UMass\\GitHub\\Image Recognition System API\\IRS_API\\IRS_API\\" + request.POST.get('url')
        img = image.load_img(my_param, target_size=(299, 299))

        plot_prediction_InceptionV3_Local(img)

        # print(my_param)

        if my_param is None:
            return render(request, 'InceptionV3_Local_Predict.html')
    return render(request, 'InceptionV3_Local_Predict.html')


def predict_InceptionV3(model_InceptionV3, img, target_size, top_n=5):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_InceptionV3.predict(x)
    return decode_predictions(preds, top=top_n)[0]


# Plotting: InceptionV3 Local Image


def plot_prediction_InceptionV3_Local(img):
    model_InceptionV3 = InceptionV3(weights='imagenet')  # Model Load
    # k = request.POST.get('url')
    prediction = predict_InceptionV3(model_InceptionV3, img, target_size=(299, 299))  # Model prediction
    prediction = pd.DataFrame(np.array(prediction).reshape(5, 3), columns=list("abc"))  # Converting to a DataFrame
    graph = prediction.convert_objects(convert_numeric=True)
    graph.to_csv('InceptionV3_Local_Graph.csv')

    print(graph)
    statement = print("This picture has the highest possibility of a " + '\033[1m' '\033[4m' + prediction.b[0])
    # print(statement)
    # display(graph)
    plt.bar(graph.b, graph.c, align='center', color='gray', edgecolor='black', width=0.4)
    plt.rcParams['figure.figsize'] = 13, 6
    plt.xlabel("Predicted Outcomes", color='blue')
    plt.ylabel("Output Probabilities(* 100 for percent)", color='blue')
    plt.savefig('IRS_API/static/InceptionV3_Local.jpg', bbox_inches='tight')
    # plt.savefig('myapp/templates/ResNet50_Local.jpg', bbox_inches='tight')s


#____________________________________________________________________________________________________________________________________

# InceptionV3_URL


def InceptionV3_URL(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'InceptionV3_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:

            return render(request, 'InceptionV3_URL.html')
    return render(request, 'InceptionV3_URL.html')

#____________________________________________________________________________________________________________________________________

 #----------------InceptionV3 Model------------------------------------------------------------------


def DenseNet201_Model(request):
    # model_VGG19 = VGG19(weights='imagenet')
    return render(request, 'DenseNet201_Model.html')


# DenseNet201_Local
def DenseNet201_Local(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'DenseNet201_Local.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'DenseNet201_Local.html')
    return render(request, 'DenseNet201_Local.html')


# DenseNet201_Local_Predict


def DenseNet201_Local_Predict(request):
    # model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('fileurl')
    # print(request.POST)
    if request.method == 'POST':
        my_param = "C:\\Users\\Kevin D'Cruz\\Downloads\\UMass\\GitHub\\Image Recognition System API\\IRS_API\\IRS_API\\" + request.POST.get('url')
        img = image.load_img(my_param, target_size=(224, 224))

        plot_prediction_DenseNet201_Local(img)

        # print(my_param)

        if my_param is None:
            return render(request, 'DenseNet201_Local_Predict.html')
    return render(request, 'DenseNet201_Local_Predict.html')


def predict_DenseNet201(model_DenseNet201, img, target_size, top_n=5):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_DenseNet201.predict(x)
    return decode_predictions(preds, top=top_n)[0]


# Plotting: DenseNet201 Local Image


def plot_prediction_DenseNet201_Local(img):
    model_DenseNet201 = DenseNet201(weights='imagenet')  # Model Load
    # k = request.POST.get('url')
    prediction = predict_DenseNet201(model_DenseNet201, img, target_size=(299, 299))  # Model prediction
    prediction = pd.DataFrame(np.array(prediction).reshape(5, 3), columns=list("abc"))  # Converting to a DataFrame
    graph = prediction.convert_objects(convert_numeric=True)
    graph.to_csv('DenseNet201_Local_Graph.csv')

    print(graph)
    statement = print("This picture has the highest possibility of a " + '\033[1m' '\033[4m' + prediction.b[0])
    # print(statement)
    # display(graph)
    plt.bar(graph.b, graph.c, align='center', color='gray', edgecolor='black', width=0.4)
    plt.rcParams['figure.figsize'] = 13, 6
    plt.xlabel("Predicted Outcomes", color='blue')
    plt.ylabel("Output Probabilities(* 100 for percent)", color='blue')
    plt.savefig('IRS_API/static/DenseNet201_Local.jpg', bbox_inches='tight')
    # plt.savefig('myapp/templates/ResNet50_Local.jpg', bbox_inches='tight')s


#____________________________________________________________________________________________________________________________________

# DenseNet201_URL


def DenseNet201_URL(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'DenseNet201_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:

            return render(request, 'DenseNet201_URL.html')
    return render(request, 'DenseNet201_URL.html')

#____________________________________________________________________________________________________________________________________
