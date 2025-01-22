import tensorflow as tf
import cv2 as cv


INPUT_DIM = (28, 28, 1)
OUTPUT_DIM = 10
GRAY_THRESHOLD = 0.0

CATEGORIES = [_ for _ in range(10)]

cnnModel = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_DIM),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_DIM, activation='softmax')
])
cnnModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnnModel.load_weights('models/mnist.h5')

'''
# Train on MNIST Dataset
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = xTrain / 255.0
xTest = xTest / 255.0

print('MNIST Train Examples:', xTrain.shape[0])
print('MNIST Test Examples:', xTest.shape[0])

#Bounding Box
imageIndex = 0
for trainImage in xTrain:
    minRow = 28
    maxRow = -1
    minCol = 28
    maxCol = -1
    for i in range(28):
        for j in range(28):
            if trainImage[i][j] > GRAY_THRESHOLD:
                minRow = min(minRow, i)
                maxRow = max(maxRow, i)
                minCol = min(minCol, j)
                maxCol = max(maxCol, j)
    trainImage = trainImage[minRow:maxRow + 1, minCol:maxCol + 1]
    trainImage = cv.resize(trainImage, (28, 28))
    trainImage = trainImage.reshape(28, 28, 1)

    print('Train Image:', imageIndex)
    imageIndex += 1



imageIndex = 0
for testImage in xTest:
    minRow = 28
    maxRow = -1
    minCol = 28
    maxCol = -1
    for i in range(28):
        for j in range(28):
            if testImage[i][j] > GRAY_THRESHOLD:
                minRow = min(minRow, i)
                maxRow = max(maxRow, i)
                minCol = min(minCol, j)
                maxCol = max(maxCol, j)
    testImage = testImage[minRow:maxRow + 1, minCol:maxCol + 1]
    testImage = cv.resize(testImage, (28, 28))
    testImage = testImage.reshape(28, 28, 1)

    print('Test Image:', imageIndex)
    imageIndex += 1



cnnModel.fit(xTrain.reshape(-1, 28, 28, 1), yTrain, epochs=5)
cnnModel.save('models/mnist.h5')
'''