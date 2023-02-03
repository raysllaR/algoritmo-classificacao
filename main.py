from keras.models import Sequential  # modelo sequencial
# Camanda de convolução, Pooling, Flatten, rede neural densa
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Normalização do mapa de caracteristicas
from keras.layers import BatchNormalization  # normalização das imagens
from keras.preprocessing.image import ImageDataGenerator  # questionavel a utilização

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

classificador.add(
    Dense(
        units=128,  # primeira camada oculta
        activation='relu'
    )
)

#
classificador.add(Dropout(0.2))  # vai zerar 20% das entradas
