import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

app = Flask(__name__)
app.template_folder = "./api/templates"

@app.route("/")
def main():
    Data = pd.read_csv(r"parkinsonsdatamodeldata.csv")

    def calculate_vif_(X, thresh=5.0):
        X = X.assign(const=1)
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                    for ix in range(X.iloc[:, variables].shape[1])]
            vif = vif[:-1]
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping \'' + X.iloc[:, variables].columns[maxloc] + '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(X.columns[variables[:-1]])
        return X.iloc[:, variables[:-1]]

    df_factors = Data.drop(['name', 'status'], axis=1)

    df = pd.DataFrame()

    df_all_factors = df_factors
    df_no_collinear = calculate_vif_(df_all_factors)

    y1 = Data['status']
    X1 = df_no_collinear

    oversample = SMOTE(random_state=9)
    X1, y1 = oversample.fit_resample(X1, y1)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1)

    ss_train = StandardScaler()
    X_train1 = ss_train.fit_transform(X_train1)

    ss_test = StandardScaler()
    X_test1 = ss_test.fit_transform(X_test1)

    rfc = RandomForestClassifier(random_state=9)
    rfc.fit(X_train1, y_train1)
    y_pred1 = rfc.predict(X_test1)

    filename = 'finalised_model1.sav'
    pickle.dump(rfc, open(filename, 'wb'))

    return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
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
        

if __name__ == "__main__":
    app.run(debug=True)
