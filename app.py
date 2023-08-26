from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            X1_transaction_date=float(request.form.get('X1 transaction date')),
            X2_house_age=float(request.form.get('X2 house age')),
            X3_distance_to_the_nearest_MRT_station=float(request.form.get('X3 distance to the nearest MRT station')),
            X4_number_of_convenience_stores=float(request.form.get('X4 number of convenience stores')),
            X5_latitude=float(request.form.get('X5 latitude')),
            X6_longitude=float(request.form.get('X6 longitude')),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        