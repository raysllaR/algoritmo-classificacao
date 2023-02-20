import keras
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from fastapi import FastAPI
from pydantic import BaseModel

def treinamento(): 
    # Pegando o arquivo de treinamento
    arquivo = open('treinamento/classificador_teste.json', 'r')

    estrutara_rede = arquivo.read()
    # Fechando o arquivo
    arquivo.close()

    # Passando o arquivo de pretreinamento para o classificador
    classificador = model_from_json(estrutara_rede)
    # Pegando os pesos
    classificador.load_weights('treinamento/classificador_teste.h5')
    
    classificador.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics='accuracy'
    )
    
    return classificador

#Initi api
app = FastAPI()

class Link(BaseModel):  
    img = []

@app.post('/classificar')
def classificationLink(link: Link):

    classificador = treinamento();
    
    resposta = {
        'Aceitas': [],
        'Recusadas': [],
        'Erros': []
        
    }
    
    for img in link.img:

        try:
            image_url = tf.keras.utils.get_file(origin= img);

            # Passar img para classificação
            imagem_teste = keras.utils.load_img(image_url, target_size=(320, 320));

            imagem_teste = keras.utils.img_to_array(imagem_teste);
            imagem_teste /= 255;
            imagem_teste = np.expand_dims(imagem_teste, axis=0);

            previsao = classificador.predict(imagem_teste);
            
            if(previsao >= 0.9):
                resposta['Aceitas'].append(img)
            else:
                resposta['Recusadas'].append(img)
            
        except:
            resposta['Erros'].append(img)

    return resposta;