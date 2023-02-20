import keras
import numpy as np
from keras.models import model_from_json

from flask import Flask, make_response

def treinamento(): 
    #Pegando o arquivo de treinamento
    arquivo = open('treinamento/classificador_teste.json', 'r')

    estrutara_rede = arquivo.read()
    #fechando o arquivo
    arquivo.close()

    # Passando o arquivo de pretreinamento para o classificador
    classificador = model_from_json(estrutara_rede)
    # Pegando os pesos
    classificador.load_weights('treinamento/classificador_teste.h5')
    
    return classificador

#Initi api
app = Flask(__name__)

@app.route('/classificar', methods=['GET'])
def classification():

    classificador = treinamento()
    #Passar img para classificação
    imagem_teste = keras.utils.load_img('../Images/images50k/AF1QipM0_6a5lWGvGO6yvp1vM9JxUiDO44XpICso4wJK=w1024-h576-k-no.jpg', target_size=(320, 320));

    imagem_teste = keras.utils.img_to_array(imagem_teste);
    imagem_teste /= 255;
    imagem_teste = np.expand_dims(imagem_teste, axis=0);

    previsao = classificador.predict(imagem_teste);

    return make_response("Aceita" if (previsao > 0.8) else "Recusada") 

app.run()
# base_treinamento.class_indices