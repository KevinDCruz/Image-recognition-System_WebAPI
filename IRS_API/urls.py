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
    url(r'^VGG19_Model$', v.VGG19_Model, name='VGG19_Model'),
    url(r'^ResNet50_Local$', v.ResNet50_Local, name='ResNet50_Local'),
    url(r'^ResNet50_URL$', v.ResNet50_URL, name='ResNet50_URL'),
    url(r'^VGG19_Local$', v.VGG19_Local, name='VGG19_Local'),
    url(r'^VGG19_URL$', v.VGG19_URL, name='VGG19_URL'),
    url(r'^ResNet50_Local_Predict$', v.ResNet50_Local_Predict, name='ResNet50_Local_Predict'),




    # url(r'^$', v.upload_local, name='process'),
]
