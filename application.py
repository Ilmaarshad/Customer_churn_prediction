from flask import Flask,request,render_template,jsonify
from ssrc.pipeline.pipeline_prediction import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('form.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            Age= float(request.form.get('Age')),
            Gender = request.form.get('Gender'),
            Location = request.form.get('Location'),
            Subscription_Length_Months = float(request.form.get('Subscription_Length_Months')),
            Monthly_Bill = float(request.form.get('Monthly_Bill')),
            Total_Usage_GB = float(request.form.get('Total_Usage_GB'))

        )
        final_new_data=data.get_data_as_dataframe()
        print("this is final new data", final_new_data)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)
        if results == 0.0:
            return render_template('result.html',final_result="NO")
        else:
            return render_template('result.html',final_result="YES")
            
        




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)