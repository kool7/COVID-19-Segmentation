#########
#IMPORTS#
#########

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
class Unet2D:
    def __init__(self):
        
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (inputs)
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(0.25)(p1)

        c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p1)
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(0.25)(p2)

        c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p2)
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(0.25)(p3)

        c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p3)
        c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(0.25)(p4)

        c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p4)
        c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c5)

        u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = BatchNormalization()(u6)
        c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u6)
        c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c6)


        u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        u7 = BatchNormalization()(u7)
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u7)
        c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c7)


        u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        u8 = BatchNormalization()(u8)
        c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u8)
        c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c8)


        u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = BatchNormalization()(u9)
        c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u9)
        c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])