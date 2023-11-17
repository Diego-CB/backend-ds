import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import json

from flask import Flask, request
app = Flask(__name__)

tokenizador = joblib.load('./tokenizador.pkl')
modelo = tf.keras.saving.load_model('./nlp_model.h5')

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
def search():
    args = request.args

    tipo_claim  = args.get('tipo')
    texto_claim = args.get('texto')

    input_encoded = procesar_texto(tipo_claim, texto_claim)
    prediccion = modelo.predict([input_encoded])
    prediccion_decoded = decode_predict(prediccion)

    return json.dumps({"prediction": prediccion_decoded})

app.run()