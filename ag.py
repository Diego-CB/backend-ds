import numpy as np
import json
import pandas as pd

from flask import Flask, request
app = Flask(__name__)

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
nombre_archivo = 'model.cpk'

# Verificar si el archivo ya existe
if verificar_existencia_archivo('./Predictor/' + nombre_archivo):
    print(f'El archivo {nombre_archivo} ya existe.')
else:
    print(f'Descargando {nombre_archivo}')
    # Descargar el archivo desde Google Drive
    descargar_modelo(file_id, nombre_archivo)
    print(f'Se ha descargado el archivo {nombre_archivo} desde Google Drive.')

shutil.move('./model.cpk', './Predictor/model.cpk')

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

    data_predict = pd.DataFrame(data_predict)
    probs = predictor.predict_proba(data_predict)[0]

    return json.dumps({
        'Adequate': probs[0],
        'Effective': probs[1],
        'Ineffective': probs[2],
    })

if __name__ == '__main__':
    app.run()