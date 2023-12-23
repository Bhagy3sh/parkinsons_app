# main.py
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        Patient_ID = request.form['Patient_ID']
        MDVP_Fo_Hz = float(request.form['MDVP_Fo_Hz'])
        MDVP_Fhi_Hz = float(request.form['MDVP_Fhi_Hz'])
        MDVP_Flo_Hz = float(request.form['MDVP_Flo_Hz'])
        MDVP_APQ = float(request.form['MDVP_APQ'])
        NHR = float(request.form['NHR'])
        RPDE = float(request.form['RPDE'])
        DFA = float(request.form['DFA'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        D2 = float(request.form['D2'])

        input_data = {'MDVP:Fo(Hz)': [MDVP_Fo_Hz], 'MDVP:Fhi(Hz)': [MDVP_Fhi_Hz], 'MDVP:Flo(Hz)': [MDVP_Flo_Hz],
                      'MDVP:APQ': [MDVP_APQ], 'NHR': [NHR], 'RPDE': [RPDE], 'DFA': [DFA],
                      'spread1': [spread1], 'spread2': [spread2], 'D2': [D2]}
        Xnew = pd.DataFrame(input_data)

        loaded_model = pickle.load(open('finalised_model1.sav', 'rb'))
        ynew = loaded_model.predict(Xnew)

        if ynew > 0.5:
            return render_template("Parkinsons.html")
        else:
            return render_template("NotParkinsons.html")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
