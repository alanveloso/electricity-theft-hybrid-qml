"""CNN para detecção de furto de energia — Pereira & Saraiva (2021), Tabela 2.

Single CNN baseline: entrada (148, 7, 1) → Conv/MaxPool × 3 → Flatten → Dense 128 → Dense 32 → Dense 2 (softmax).
O modelo é retornado por build_cnn() para permitir remover a última camada e conectar o circuito quântico.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def build_cnn(input_shape=(148, 7, 1)):
    """
    Constrói a Single CNN exatamente conforme Tabela 2 do artigo (Pereira & Saraiva, 2021).

    Arquitetura:
    - Input (148, 7, 1)
    - Conv2D 16, (3,3), same, ReLU
    - MaxPooling2D (4, 1) → redução 148→37
    - Conv2D 32, (3,3), same, ReLU
    - MaxPooling2D (4, 1) → redução 37→9
    - Conv2D 64, (3,3), same, ReLU
    - MaxPooling2D (3, 2) → redução final
    - Flatten → Dense 128 ReLU → Dense 32 ReLU → Dense 2 Softmax

    Retorna o modelo (Keras Sequential) para uso como baseline ou para remover
    a última camada e conectar o classificador quântico.
    """
    model = Sequential(name="single_cnn_pereira")

    # Input + Camada 1: Conv2D 16
    model.add(
        Conv2D(
            16,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
            name="conv2d_16",
        )
    )
    # Camada 2: MaxPooling2D → (37, 7, 16) com pool (4, 1)
    model.add(MaxPooling2D(pool_size=(4, 1), padding="same", name="maxpool_1"))

    # Camada 3: Conv2D 32
    model.add(
        Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_32")
    )
    # Camada 4: MaxPooling2D → (9, 7, 32)
    model.add(MaxPooling2D(pool_size=(4, 1), padding="same", name="maxpool_2"))

    # Camada 5: Conv2D 64
    model.add(
        Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_64")
    )
    # Camada 6: MaxPooling2D → (3, 3 ou 4, 64)
    model.add(MaxPooling2D(pool_size=(3, 2), padding="same", name="maxpool_3"))

    # Camada 7: Flatten
    model.add(Flatten(name="flatten"))

    # Camadas 8–9: FC 128 e 32
    model.add(Dense(128, activation="relu", name="dense_128"))
    model.add(Dense(32, activation="relu", name="dense_32"))

    # Saída: 2 classes (Normal vs Fraude)
    model.add(Dense(2, activation="softmax", name="output"))

    return model


# Alias para compatibilidade com código que já usa build_paper_cnn
def build_paper_cnn(input_shape=(148, 7, 1)):
    """Alias para build_cnn (Pereira & Saraiva, 2021)."""
    return build_cnn(input_shape=input_shape)
