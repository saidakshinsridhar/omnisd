import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

# --- 1. Define Paths and Parameters ---
# Make sure this path matches the folder you downloaded
DATA_DIR = 'data/dataset-resized' 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. Load and Augment Data ---
# This automatically loads images from subfolders and labels them
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to be between 0 and 1
    validation_split=0.2     # Use 20% of the data for validation
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- 3. Build the Model ---
# Load a powerful pre-trained model (MobileNetV2) without its top classification layer
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model layers so we don't retrain them
base_model.trainable = False

# Add our own custom classification layers on top
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
# The final layer has one output neuron for each class (cardboard, glass, etc.)
predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile and Train ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for 5 epochs (you can increase this later if you have time)
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

# --- 5. Save the Final Model ---
# This is the most important step!
model.save('backend/models/trash_classifier.h5')

print("\n✅ Model training complete and saved to backend/models/trash_classifier.h5")