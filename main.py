from keras.models import Sequential  # modelo sequencial
# Camanda de convolução, Pooling, Flatten, rede neural densa
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Normalização do mapa de caracteristicas
from keras.layers import BatchNormalization  # normalização das imagens
from keras.preprocessing.image import ImageDataGenerator  # questionavel a utilização

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

classificador = Sequential()


# primeira camada de convolução
classificador.add(
    Conv2D(
        64,  # número de filtro
        (9, 9),  # dimensão do detectores de caracteristicas
        # Altura e largura da img e numero de canais (3 significa RGB) - converção das imagens originais para este padrão
        input_shape=(320, 320, 3),
        activation='relu'  # função de ativação - tirar os valores negativos, que representam as partes mais escuras das imagens
    )
)

classificador.add(
    BatchNormalization()  # deixa os valores do mapa de caracteristica em uma escala de 0 e 1
)

classificador.add(
    MaxPooling2D(
        # matriz de 4 pixels pegando as caracteristicas mais importantes
        pool_size=(2, 2)
    )
)

# segunda camada de convolução
classificador.add(
    Conv2D(
        64,  # número de filtro
        (9, 9),  # dimensão do detectores de caracteristicas
        # Altura e largura da img e numero de canais (3 significa RGB) - converção das imagens originais para este padrão
        input_shape=(320, 320, 3),
        activation='relu'  # função de ativação - tirar os valores negativos, que representam as partes mais escuras das imagens
    ))

classificador.add(
    BatchNormalization()  # deixa os valores do mapa de caracteristica em uma escala de 0 e 1
)

classificador.add(
    MaxPooling2D(
        # matriz de 4 pixels pegando as caracteristicas mais importantes
        pool_size=(2, 2)
    )
)

# Flattening

classificador.add(Flatten())  # transforma a matriz em vetor

# Rede neural densa


# primeira camada oculta

classificador.add(
    Dense(
        units=128,
        activation='relu'
    )
)

classificador.add(Dropout(0.2))  # vai zerar 20% das entradas

# segunda camada oculta

classificador.add(
    Dense(
        units=128,
        activation='relu'
    )
)

classificador.add(Dropout(0.2))  # vai zerar 20% das entradas

# saída
classificador.add(
    Dense(
        units=1,  # classificação binaria - aceita ou negada
        activation='sigmoid'
    )
)

# compile
classificador.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics='accuracy'
)

# normalização dos dados - treinamento
gerador_treinamento = ImageDataGenerator(
    rescale=1./255,
    rotation_range=7,
    horizontal_flip=True,  # vai fazer giros horizontais nas imagens
    shear_range=0.2,
    height_shift_range=0.07,
    zoom_range=0.2
)

# gerador teste
gerador_teste = ImageDataGenerator(rescale=1./255)

# base treinamento
base_treinamento = gerador_treinamento.flow_from_directory(
    'dataset/training',
    target_size=(320, 320),  # tamanho das imagens
    batch_size=16,
    class_mode='binary'  # como vai ficar o problema de classificação
)

# base teste
base_teste = gerador_teste.flow_from_directory(
    'dataset/test',
    target_size=(320, 320),  # tamanho das imagens
    batch_size=32,
    class_mode='binary'  # como vai ficar o problema de classificação
)

# executar treinamento
classificador.fit(
    base_treinamento,
    steps_per_epoch=2000/16,
    epochs=8,
    validation_data=base_teste,
    validation_steps=2000/16
)

#Salvar a classificação


classificador_json = classificador.to_json()
with open('treinamento/classificador_teste.json', 'w') as json_file:
    json_file.write(classificador_json)
    
#Salvando os pesos

classificador.save_weights('treinamento/classificador_teste.h5') # pip install h5py