import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from dataset import load_dataset

sys.path.append(os.pardir)
(train_data, train_labels), (valid_data, valid_label) = load_dataset(flatten = True, normalize = False)
train_data, valid_data = train_data / 255.0, valid_data / 255.0

class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1, use_downsample=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        if use_downsample:
            self.downsample = layers.Conv2D(32, kernel_size=1, strides=strides, use_bias=False)
        else:
            self.downsample = lambda x: x
        
    def call(self, inputs):
        residual = self.downsample(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        output = self.relu(x + residual)
        
        return output

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(4, 800, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    ResidualBlock(32),
    ResidualBlock(32),
    ResidualBlock(32),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    ResidualBlock(64),
    ResidualBlock(64),
    ResidualBlock(64),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(18, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_data, train_labels, epochs=25, validation_data=(valid_data, valid_label))
test_loss, test_acc = model.evaluate(valid_data, valid_label)
print('Test accuracy:', test_acc)

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
