from flask import Flask,render_template,request
import joblib
import numpy as np
app=Flask(__name__)

model=joblib.load("model.pkl")
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/result",methods=["GET","POST"])
def result():
    if request.method == "POST":
        result=request.form
        features=[float(x) for x in result.values()]
        final_features=[np.array(features)]
        res=model.predict(final_features)
        return render_template("result.html",result=res)
if __name__ == __name__:
    app.run(debug=True)