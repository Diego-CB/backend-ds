import numpy as np
import json
import pandas as pd
from flask_cors import CORS

from flask import Flask, request
app = Flask(__name__)
CORS(app)

## Descargar modelo
import os
import gdown
import shutil

def descargar_modelo(file_id, nombre_archivo):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = nombre_archivo
    gdown.download(url, output, quiet=False)

def verificar_existencia_archivo(nombre_archivo):
    return os.path.isfile(nombre_archivo)

# Definir el ID del archivo en Google Drive y el nombre del archivo
file_id = '1zgFGu-WuVzex8jXOTYSnfMpGGVxAQ5pu'
nombre_archivo = 'model.ckpt'

# Verificar si el archivo ya existe
if verificar_existencia_archivo('Predictor/' + nombre_archivo) or verificar_existencia_archivo(nombre_archivo):
    print(f'El archivo {nombre_archivo} ya existe.')
else:
    print(f'Descargando {nombre_archivo}')
    # Descargar el archivo desde Google Drive
    descargar_modelo(file_id, nombre_archivo)
    print(f'Se ha descargado el archivo {nombre_archivo} desde Google Drive.')

try:
    shutil.move('./model.cpk', './Predictor/model.ckpt')

except:
    pass

from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor.load('Predictor')

@app.route('/', methods=['GET'])
def home():
    return json.dumps({"Server": 'Running'})

@app.route('/predict', methods=['GET'])
def search():
    print('> entro API')
    args = request.args

    tipo_claim  = args.get('tipo')
    texto_claim = args.get('texto')

    if None in [tipo_claim, texto_claim]:
        return json.dumps({"prediction": 'not enough params'})

    print('> params:', tipo_claim, texto_claim)

    data_predict = {
        'discourse_text': [texto_claim],
        'discourse_type': [tipo_claim],
        'claim_size': [len(texto_claim)],
    }
    print(data_predict)

    data_predict = pd.DataFrame(data_predict)
    print(data_predict)
    probs = predictor.predict_proba(data_predict)
    print('-----------------------')
    print(probs)
    print('-----------------------')
    print(probs['Adequate'].values)
    print(probs['Effective'].values)
    print(probs['Ineffective'].values)

    return json.dumps({
        'Adequate': str(probs['Adequate'].values[0]),
        'Effective': str(probs['Effective'].values[0]),
        'Ineffective': str(probs['Ineffective'].values[0]),
    })

if __name__ == '__main__':
    app.run()
