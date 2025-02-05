import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from model import resnet34

# Define paths
data_dir = "dataset"
train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "test")

# Data augmentation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=10,
    height_shift_range=10,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)
test_data_gen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_data_gen.flow_from_directory(
    train_path, target_size=(48, 48), batch_size=32, color_mode='grayscale', class_mode='categorical', shuffle=True, subset='training'
)
val_data = train_data_gen.flow_from_directory(
    train_path, target_size=(48, 48), batch_size=32, color_mode='grayscale', class_mode='categorical', shuffle=True, subset='validation'
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(train_data.classes), y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Model training
model = resnet34()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'], loss='categorical_crossentropy')

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(train_data, validation_data=val_data, epochs=300, callbacks=[early_stop, model_checkpoint], class_weight=class_weights)

print("Model training complete.")

