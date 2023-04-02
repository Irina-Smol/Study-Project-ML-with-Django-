from django.shortcuts import render, HttpResponse
import os
from django.http import JsonResponse
import joblib
import numpy as np
import pandas as pd

def index(request):
    return render(request, 'index.html')

def result(request):
    data = joblib.load('final_modeling.sav')
    regressor = data['regressor']
    vectorizer = data['vectorizer']


    sound = request.GET.get('sound')
    #sound = pd.reshape(sound, (-1, 1))
    #sound = sound.astype('float64')
    sound = pd.Series([sound])
    sound = sound.str()

            # vectorize sound
    vector = vectorizer.transform({sound})
    vector = vector.astype('float64')
            # predict based on vector
    prediction = regressor.predict(vector[0])
            # build response
    result = {'dog': prediction}
            # return response
    return render(request, 'result.html', {'result': result})
