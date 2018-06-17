"""IRS_API URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from myapp import views as v


urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^$', v.index, name='indexpage'),
    url(r'^Models$', v.Models, name='Models'),
    url(r'^ResNet50_Model$', v.ResNet50_Model, name='ResNet50_Model'),
    url(r'^ResNet50_Local$', v.ResNet50_Local, name='ResNet50_Local'),
    url(r'^ResNet50_URL$', v.ResNet50_URL, name='ResNet50_URL'),
    url(r'^ResNet50_Local_Predict$', v.ResNet50_Local_Predict, name='ResNet50_Local_Predict'),

    url(r'^VGG19_Model$', v.VGG19_Model, name='VGG19_Model'),
    url(r'^VGG19_Local$', v.VGG19_Local, name='VGG19_Local'),
    url(r'^VGG19_URL$', v.VGG19_URL, name='VGG19_URL'),
    url(r'^VGG19_Local_Predict$', v.VGG19_Local_Predict, name='VGG19_Local_Predict'),

    url(r'^VGG16_Model$', v.VGG16_Model, name='VGG16_Model'),
    url(r'^VGG16_Local$', v.VGG16_Local, name='VGG16_Local'),
    url(r'^VGG16_URL$', v.VGG16_URL, name='VGG16_URL'),
    url(r'^VGG16_Local_Predict$', v.VGG16_Local_Predict, name='VGG16_Local_Predict'),

    url(r'^InceptionV3_Model$', v.InceptionV3_Model, name='InceptionV3_Model'),
    url(r'^InceptionV3_Local$', v.InceptionV3_Local, name='InceptionV3_Local'),
    url(r'^InceptionV3_URL$', v.InceptionV3_URL, name='InceptionV3_URL'),
    url(r'^InceptionV3_Local_Predict$', v.InceptionV3_Local_Predict, name='InceptionV3_Local_Predict'),

    url(r'^DenseNet201_Model$', v.DenseNet201_Model, name='DenseNet201_Model'),
    url(r'^DenseNet201_Local$', v.DenseNet201_Local, name='DenseNet201_Local'),
    url(r'^DenseNet201_URL$', v.DenseNet201_URL, name='DenseNet201_URL'),
    url(r'^DenseNet201_Local_Predict$', v.DenseNet201_Local_Predict, name='DenseNet201_Local_Predict'),



    # url(r'^$', v.upload_local, name='process'),
]
