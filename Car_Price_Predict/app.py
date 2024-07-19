import json
from flask import Flask, render_template, request , jsonify
import pickle
import pandas as pd

with open('brand_mean_price.json', 'r') as f:
    brand_mean_price = json.load(f)

with open('model_mean_price.json', 'r') as f:
    model_mean_price = json.load(f)

with open('marka_model_dict.json', 'r') as f:
    marka_model_dict = json.load(f)


app = Flask(__name__)

model = pickle.load(open('CarPrediction_Model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', marka_model_dict=marka_model_dict)
    
@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form['Marka']
    model_name = request.form['Model']
    km = int(request.form['Km'])
    color = request.form['Renk']
    age = int(request.form['Yaş'])
    gear = request.form['Vites']
    fuel = request.form['Yakıt']

    car_data = {
        'Marka': brand,
        'Model': model_name,
        'Km': km,
        'Renk': color,
        'Yaş': age,
        'Vites': gear,
        'Yakıt': fuel,
    }

    input_data = pd.DataFrame([car_data])

    required_columns = [
    'Marka', 'Model', 'Km', 'Yaş', 
    'Renk_Bej','Renk_Beyaz', 'Renk_Bordo', 'Renk_Füme', 'Renk_Gri', 'Renk_Gümüş Gri', 'Renk_Kahverengi', 'Renk_Kırmızı','Renk_Lacivert' ,'Renk_Mavi','Renk_Mor', 'Renk_Pembe', 'Renk_Sarı', 'Renk_Siyah','Renk_Turkuaz', 'Renk_Turuncu', 'Renk_Yeşil',
    'Renk_Şampanya',
    'Vites_Manuel','Vites_Otomatik', 'Vites_Yarı Otomatik',
    'Yakıt_Benzin','Yakıt_Benzin & LPG','Yakıt_Dizel', 'Yakıt_Elektrik', 'Yakıt_Hybrid',
    'Fiyat'
    ]

    input_data = pd.get_dummies(input_data, columns= ['Renk', 'Vites', 'Yakıt'],dtype=int)

    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[required_columns]
    input_data['Brand_Encoded'] = brand_mean_price.get(brand, 0)
    input_data['Model_Encoded'] = model_mean_price.get(model_name, 0)
    input_data = input_data.drop(["Fiyat","Marka","Model"],axis=1)

    prediction = model.predict(input_data)[0]

    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)