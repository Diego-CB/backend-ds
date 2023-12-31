import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import json
from flask_cors import CORS

from flask import Flask, request
app = Flask(__name__)
CORS(app)

tokenizador = joblib.load('./tokenizador.pkl')

## Descargar modelo
import os
import gdown

def descargar_modelo(file_id, nombre_archivo):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = nombre_archivo
    gdown.download(url, output, quiet=False)

def verificar_existencia_archivo(nombre_archivo):
    return os.path.isfile(nombre_archivo)

# Definir el ID del archivo en Google Drive y el nombre del archivo
file_id = '19R31Yz132zyAdNVO9gW0KvjEvXqsjut5'
nombre_archivo = 'nlp_model.h5'

# Verificar si el archivo ya existe
if verificar_existencia_archivo(nombre_archivo):
    print(f'El archivo {nombre_archivo} ya existe.')
else:
    print(f'Descargando {nombre_archivo}')
    # Descargar el archivo desde Google Drive
    descargar_modelo(file_id, nombre_archivo)
    print(f'Se ha descargado el archivo {nombre_archivo} desde Google Drive.')


modelo = tf.keras.saving.load_model('./nlp_model.h5')

# procesar texto

def procesar_texto(claim:str, texto: str):
    ''' Necesita el Tokenizador creado arriba '''

    # Tokenizar las frase
    secuencia = tokenizador.texts_to_sequences(texto)
    secuencia = np.array(secuencia).T[0]

    # Rellenar (Pad) las secuencias para que tengan la misma longitud
    to_pad = np.array([[0 for _ in range(846)], secuencia.tolist()], dtype='object')
    secuencias = pad_sequences(to_pad)
    secuencia = secuencias[1]
    tensor_secuencias = tf.stack([secuencia])
    claim_size = len(texto)

    # Se hace el encoding del tipo de claiim
    claim_map = {'Lead': 0, 'Position': 1, 'Claim': 2, 'Evidence': 3, 'Counterclaim': 4, 'Rebuttal': 5, 'Concluding Statement': 6}

    if claim not in claim_map.keys():
        raise Exception(f'tipo de argumento \'{claim}\' no aceptado')

    # Preparar inputs para los 2 pipelines
    encoded_claim = claim_map[claim]
    X = np.array([[encoded_claim, claim_size]])

    # Juntar ambos pipeliins
    input_encoded = [tensor_secuencias, X]

    return input_encoded

def decode_predict(predict):
    decode_map = ['Adequate', 'Effective', 'Ineffective']
    return decode_map[int(predict)]

@app.route('/', methods=['GET'])
def home():
    return json.dumps({"Server": 'Running'})

@app.route('/predict', methods=['GET'])
def search():
    try:
        print('> entro API')
        args = request.args

        tipo_claim  = args.get('tipo')
        texto_claim = args.get('texto')

        if None in [tipo_claim, texto_claim]:
            return json.dumps({"prediction": 'not enough params'})

        print('> params:', tipo_claim, texto_claim)

        print('> Procesando input')
        input_encoded = procesar_texto(tipo_claim, texto_claim)
        print('> Predict')
        prediccion = modelo.predict([input_encoded])
        print('> Decoded predict')
        prediccion_decoded = decode_predict(prediccion)
        print('> Sending')

        adequate = '1' if prediccion_decoded == 'Adequate' else '0'
        Effective = '1' if prediccion_decoded == 'Effective' else '0'
        Ineffective = '1' if prediccion_decoded == 'Ineffective' else '0'

        return json.dumps({
            'Adequate': adequate,
            'Effective': Effective,
            'Ineffective': Ineffective,
        })
    except:
        return json.dumps({
            'Adequate': 0,
            'Effective': 0,
            'Ineffective': 1,
        })


if __name__ == '__main__':
    app.run(port=5001)