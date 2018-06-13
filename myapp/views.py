from django.shortcuts import render, render_to_response
from tkinter import filedialog
from tkinter import *
from django.core.files.storage import FileSystemStorage
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
        result = 10.5
        return render(request, "index.html", {'result': result})


def upload_local(request, localimage):
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    print(root.filename)
    return
    # render_to_response("index.html", RequestContext(request, {}))

# upload_local()


def ResNet50(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'ResNet50.html', {'uploaded_file_url': "static/" + myfile.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'ResNet50.html')
    return render(request, 'ResNet50.html')


def VGG19(request):
    if request.method == 'POST' and request.FILES['myfile2']:
        myfile2 = request.FILES['myfile2']
        fs = FileSystemStorage()
        filename = fs.save("IRS_API/static/" + myfile2.name, myfile2)
        uploaded_file_url = fs.url(filename)
        return render(request, 'VGG19.html', {'uploaded_file_url': "static/" + myfile2.name})
    print(request.GET)
    if request.method == 'GET':
        my_param = request.GET.get('run')
        if my_param is None:
            return render(request, 'VGG19.html')
    return render(request, 'VGG19.html')
