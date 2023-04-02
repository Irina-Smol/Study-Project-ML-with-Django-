from django.urls import path, re_path
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('/res', result, name='result'),
]


