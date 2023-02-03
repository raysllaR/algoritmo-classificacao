from keras.models import Sequential  # modelo sequencial
# Camanda de convolução, Pooling, Flatten, rede neoral densa
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
# Normalização do mapa de caracteristicas
from keras.layers import BatchNormalization  # normalização das imagens
from keras.preprocessing.image import ImageDataGenerator  # questionavel a utilização

classificador = Sequential()

classificador.add(
    Conv2D(  # primeira camada de convolução
        64,  # número de filtro
        (9, 9),  # dimensão do detectores de caracteristicas
        # Altura e largura da img e numero de canais (3 significa RGB) - converção das imagens originais para este padrão
        input_shape=(320, 320, 3),
        activation='relu'  # função de ativação - tirar os valores negativos, que representam as partes mais escuras das imagens
    ))

classificador.add(
    BatchNormalization()  # deixa os valores do mapa de caracteristica em uma escala de 0 e 1
)
