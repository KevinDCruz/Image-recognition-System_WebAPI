"""import django
print(django.get_version())
import numpy
print(numpy.version())
"""
from tkinter import filedialog
from tkinter import *


def upload_local():

    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    return root.filename


#upload_local()
