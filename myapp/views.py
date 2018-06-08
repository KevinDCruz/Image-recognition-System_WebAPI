from django.shortcuts import render, render_to_response
from tkinter import filedialog
from tkinter import *

# Create your views here.


def index(request):
    return render_to_response('index.html')


def upload_local():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

    print(root.filename)
    render_to_response("index.html", RequestContext(request, {}))

# upload_local()
