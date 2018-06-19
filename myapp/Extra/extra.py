

#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

#----------------VGG16 Model------------------------------------------------------------------
"""

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

"""


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
"""
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
"""

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


# ResNet50_URL


def ResNet50_URL(request):
    if request.method == 'POST' and request.POST.get('myfile'):
        print("Howdy")
        # myfile = request.FILES['myfile']
        # filename = urllib.request.urlretrieve(myfile, "Hey.jpg")
        myfile = request.POST.get('myfile')
        #myfile = myfile.replace("https://", "http://")
        #myfile_format = '"%s"' % myfile
        url_response = urllib.request.urlopen(myfile)  # extract the contents of the URL
        # image_url = image.load_img(url_response, target_size=(224, 224))  # img = image.load_img(img, target_size=(224, 224))
        #img = urllib.urlretrieve(myfile_format, "abc.jpg")
        #img = urllib.request.urlretrieve(myfile_format, "abc.jpg")
        print(myfile)
        return render(request, 'ResNet50_URL.html')
    else:
        return render(request, 'ResNet50_URL.html')


"""
    if request.method == 'POST':
        print()  # and request.FILES['myfile']:
        filename = urllib.request.urlretrieve("myfile")
        print(filename)
        # filename = urllib.request. ("myfile", "IRS_API/static/local-filename.jpg")  # extract the contents of the URL
        uploaded_file_url = fs.url(filename)
        return render(request, 'ResNet50_URL.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'ResNet50_URL.html') """


# myfile = image.load_img(url_response, target_size=(224, 224))  # img = image.load_img(img, target_size=(224, 224))
"""
    # urllib.request.urlretrieve('myfile')
    # myfile = request.FILES['myfile']
    # fs = FileSystemStorage()
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


def ResNet50_URL_Predict(request):
    # model_ResNet50 = ResNet50(weights='imagenet')  # Model Load
    # k = request.POST.get('fileurl')
    # print(request.POST)
    if request.method == 'POST':
        img = image.load_img(myfile, target_size=(224, 224))

        plot_prediction(img)

        # print(my_param)

        if my_param is None:
            return render(request, 'ResNet50_URL_Predict.html')
    return render(request, 'ResNet50_URL_Predict.html')
    """
    k = request.GET.get('uploaded_file_url')
    print(k)
    return render(request, 'ResNet50_Local_Predict.html')


    plot_prediction()
    return render(request, 'ResNet50_Local_Predict.html')
    """


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


#____________________________________________________________________________________________________________________________________
