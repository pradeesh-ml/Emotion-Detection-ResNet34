import tensorflow as tf

def identity_block(x, filters):
    x_skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x

def convolutional_block(x, filters):
    x_skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x_skip = tf.keras.layers.Conv2D(filters, (1, 1), strides=2, padding='same')(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x

def resnet34(input_shape=(48, 48, 1), classes=7):
    x_input = tf.keras.layers.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=1)(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = convolutional_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    
    x = convolutional_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    
    x = convolutional_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    
    x = convolutional_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name='ResNet34')
    return model

