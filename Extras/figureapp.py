import csv
import pandas as pd
import matplotlib.pyplot as plt
import string
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import numpy as np


data_pred= pd.read_csv("predictions.csv")
sizes = data_pred['Emotion'].value_counts()
sizes = np.array(sizes)
print(sizes)
labels = np.array(data_pred['Emotion'].unique())
def pieC():
      data_pred= pd.read_csv("predictions.csv")
      sizes = data_pred['Emotion'].value_counts()
      sizes = np.array(sizes)
      print(sizes)
      labels = data_pred['Emotion'].unique()
      plt.figure(figsize=(8,10))
      plt.pie(sizes, labels=labels,
        autopct='%.2f%%', shadow=True, startangle=100)
      plt.axis('equal')
      
      # plt.legend(labels=labels, loc='upper center', 
      #      bbox_to_anchor=(0.5, -0.04), ncol=2)
      #plt.show()

      #cn[:].plot(kind='pie', title='Visualization of Sentiment Emotions',autopct='%.2f%%',pctdistance=0.6,shadow=True)

      plt.savefig("Outputs/output_barchart.jpg")
      print("Saved Image")
def barC():
      data_pred['Emotion'].value_counts().plot(kind="bar",figsize=(7, 6),rot=0)
      plt.xlabel("Emotions", labelpad=14)
      plt.ylabel("Emotion Count", labelpad=14)
      plt.title("Count of People Who Received Tips by Gender", y=1.02)
      plt.savefig("Outputs/output_bar.jpg")
barC()
app = FastAPI()
@app.get("/")
def read_root():
    return {"Emotela": ":Say Hi!"}
@app.get("/barchart")
async def barchart():
    img = "Outputs/output.jpg"
    return FileResponse(img)
    return Response(content=img, media_type="image/jpeg")