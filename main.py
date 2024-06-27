import pygame as pg
import numpy as np
import sklearn as skl
import sklearn.datasets as datasets
import pandas as pd
import sklearn.naive_bayes as naive_bayes
import sklearn.neural_network as neural_network

# Model

trainSize = 10000

mnist = datasets.fetch_openml('mnist_784', version=1, cache=True)

trainImages = np.array(mnist.data.astype(np.uint8)).reshape((-1, 28 * 28))[:trainSize]
trainLabels = np.array(mnist.target.astype(np.uint8))[:trainSize]

randomPermutation = np.random.permutation(trainSize)

trainImages = trainImages[randomPermutation]
trainLabels = trainLabels[randomPermutation]

# KNN
def predictKNN(testImage, trainImages, trainLabels, K=100, metric='l2'):
    if metric == 'l1':
        distances = np.sum(np.abs(trainImages - testImage), axis=1)
    elif metric == 'l2':
        distances = np.sqrt(np.sum((trainImages - testImage) ** 2, axis=1))
    indices = distances.argsort()
    bestKIndices = indices[:K]
    bestKLabels = trainLabels[bestKIndices]
    return np.bincount(bestKLabels)


'''
# Naive Bayes
numBins = 4
# probability [label][pixel pos][bin index] / total tests
naiveBayesProbabilities = [[[0.0] * numBins for j in range(28 * 28)] for i in range(10)]

def initNaiveBayes(trainImages, trainLabels):
    global naiveBayesProbabilities
    global numBins
    for image, lab in zip(trainImages, trainLabels):
        for pixelPos in range(image.shape[0]):
            binIndex = image[pixelPos] // (256 // numBins)
            naiveBayesProbabilities[lab][pixelPos][binIndex] += 1.0
    for lab in range(10):
        for pixelPos in range(28 * 28):
            for binIndex in range(numBins):
                naiveBayesProbabilities[lab][pixelPos][binIndex] /= trainImages.shape[0]
                if naiveBayesProbabilities[lab][pixelPos][binIndex] != 0:
                    naiveBayesProbabilities[lab][pixelPos][binIndex] = np.log(naiveBayesProbabilities[lab][pixelPos][binIndex])


initNaiveBayes(trainImages, trainLabels)

def predictNaiveBayes(testImage):
    global numBins
    global naiveBayesProbabilities
    naiveBayesProbs = [0.0] * 10
    for lab in range(10):
        for pixelPos in range(testImage.shape[0]):
            binIndex = testImage[pixelPos] // (256 // numBins)
            naiveBayesProbs[lab] += naiveBayesProbabilities[lab][pixelPos][binIndex]
    return np.array(naiveBayesProbs)

multinomialNB = naive_bayes.MultinomialNB()
multinomialNB.fit(trainImages, trainLabels)

# MLP Classifier

mlpClassifier = neural_network.MLPClassifier(hidden_layer_sizes=(64, 64), alpha=0.001, early_stopping=True)
mlpClassifier.fit(trainImages, trainLabels)
'''


# Interface, PyGame

pg.init()

pixelWidth = 25
pixelHeight = 25

numPixelsDrawWidth = 28
numPixelsHeight = 28

drawWidth = numPixelsDrawWidth * pixelWidth
drawHeight = numPixelsHeight * pixelHeight

numPixelsPredictionWidth = 28

predictionWidth = numPixelsPredictionWidth * pixelWidth

screenWidth = drawWidth + predictionWidth
screenHeight = drawHeight

fontSize = 40
font = pg.font.Font(None, fontSize)

screen = pg.display.set_mode((screenWidth, screenHeight))

# clear
screen.fill((0, 0, 0))
drawMatrix = [[0] * numPixelsDrawWidth for y in range(numPixelsHeight)]
drawCells = [[pg.Rect(x * pixelWidth, y * pixelHeight, pixelWidth, pixelHeight) for x in range(numPixelsDrawWidth)] for y in range(numPixelsHeight)]  # x, y, width, height

# brush parameters
drawSpeed = 5
drawRadius = 2

clearSpeed = 25
clearRadius = 3

pg.display.set_caption("DigitRecognizer")
icon = pg.image.load("assets/sprites/iconDigit.png")
pg.display.set_icon(icon)

isRunning = True
isHoldingLeftClick = False
isHoldingRightClick = False


currentTime = pg.time.get_ticks()
previousTime = currentTime
currentTime = pg.time.get_ticks()
deltaTime = currentTime - previousTime


while isRunning:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            isRunning = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == pg.BUTTON_LEFT:
                isHoldingLeftClick = True
            elif event.button == pg.BUTTON_RIGHT:
                isHoldingRightClick = True
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == pg.BUTTON_LEFT:
                isHoldingLeftClick = False
            elif event.button == pg.BUTTON_RIGHT:
                isHoldingRightClick = False
    if isHoldingLeftClick:
        (mouseX, mouseY) = pg.mouse.get_pos()
        if 0 <= mouseX < screenWidth and 0 <= mouseY < screenHeight:
            if mouseX < drawWidth:
                drawMatrixLine = mouseY // pixelHeight
                drawMatrixColumn = mouseX // pixelWidth
                drawMatrix[drawMatrixLine][drawMatrixColumn] = min(255, drawMatrix[drawMatrixLine][drawMatrixColumn] + drawSpeed * deltaTime)
                for lin in range(drawMatrixLine - drawRadius + 1, drawMatrixLine + drawRadius):
                    for col in range(drawMatrixColumn - drawRadius + 1, drawMatrixColumn + drawRadius):
                        if lin < 0 or lin >= numPixelsHeight or col < 0 or col >= numPixelsDrawWidth:
                            continue
                        dist = abs(drawMatrixLine - lin) + abs(drawMatrixColumn - col)
                        if dist == 0:  # or dist >= drawRadius:
                            continue
                        drawMatrix[lin][col] = min(255, drawMatrix[lin][col] + drawSpeed * deltaTime // dist)

    if isHoldingRightClick:
        (mouseX, mouseY) = pg.mouse.get_pos()
        if 0 <= mouseX < screenWidth and 0 <= mouseY < screenHeight:
            if mouseX < drawWidth:
                drawMatrixLine = mouseY // pixelHeight
                drawMatrixColumn = mouseX // pixelWidth
                drawMatrix[drawMatrixLine][drawMatrixColumn] = max(0, drawMatrix[drawMatrixLine][drawMatrixColumn] - clearSpeed * deltaTime)
                for lin in range(drawMatrixLine - clearRadius + 1, drawMatrixLine + clearRadius):
                    for col in range(drawMatrixColumn - clearRadius + 1, drawMatrixColumn + clearRadius):
                        if lin < 0 or lin >= numPixelsHeight or col < 0 or col >= numPixelsDrawWidth:
                            continue
                        dist = abs(drawMatrixLine - lin) + abs(drawMatrixColumn - col)
                        if dist == 0:  # or dist >= clearRadius:
                            continue
                        drawMatrix[lin][col] = max(0, drawMatrix[drawMatrixLine][drawMatrixColumn] - clearSpeed * deltaTime // dist)

    clearPredictionScreen = pg.Rect(drawWidth, 0, predictionWidth, screenHeight)
    pg.draw.rect(screen, (50, 50, 50), clearPredictionScreen)

    for i in range(len(drawCells)):
        for j in range(len(drawCells[i])):
            luminosity = drawMatrix[i][j]
            pg.draw.rect(screen, (luminosity, luminosity, luminosity), drawCells[i][j])

    # KNN
    binCountsKNN = predictKNN(np.array(drawMatrix).reshape((-1,)), trainImages, trainLabels)

    for label, frequency in enumerate(binCountsKNN):
        textToRender = str(label) + ': ' + str(round(100 * frequency / np.sum(binCountsKNN), 2)) + '%'
        if label == np.argmax(binCountsKNN):
            textSurface = font.render(textToRender, True, (0, 255, 0))
        else:
            textSurface = font.render(textToRender, True, (255, 255, 255))
        screen.blit(textSurface, (drawWidth + predictionWidth // 2, label * screenHeight // 10 + fontSize // 2))
    for label in range(len(binCountsKNN), 10):
        textToRender = str(label) + ': ' + str(round(0, 2)) + '%'
        textSurface = font.render(textToRender, True, (255, 255, 255))
        screen.blit(textSurface, (drawWidth + predictionWidth // 2, label * screenHeight // 10 + fontSize // 2))

    '''
    # Naive Bayes
    naiveBayesProb = predictNaiveBayes(np.array(drawMatrix).reshape((-1,)))

    for label, prob in enumerate(naiveBayesProb):
        if np.sum(naiveBayesProb) == 0.0:
            probSum = 1.0
        else:
            probSum = np.sum(naiveBayesProb)
        textToRender = str(label) + ': ' + str(round(100 * prob / probSum, 2)) + '%'
        if label == np.argmax(naiveBayesProb):
            textSurface = font.render(textToRender, True, (0, 255, 0))
        else:
            textSurface = font.render(textToRender, True, (255, 255, 255))
        screen.blit(textSurface, (drawWidth + predictionWidth // 2, label * screenHeight // 10 + fontSize // 2))

    print(multinomialNB.predict(np.array(drawMatrix).reshape((1, 784))))
    '''

    '''
    mlpPredictions = mlpClassifier.predict_proba(np.array(drawMatrix).reshape((1, -1))).reshape((-1, ))

    for label, prob in enumerate(mlpPredictions):
        textToRender = str(label) + ': ' + str(round(100 * prob / np.sum(mlpPredictions), 2)) + '%'
        if label == np.argmax(mlpPredictions):
            textSurface = font.render(textToRender, True, (0, 255, 0))
        else:
            textSurface = font.render(textToRender, True, (255, 255, 255))
        screen.blit(textSurface, (drawWidth + predictionWidth // 2, label * screenHeight // 10 + fontSize // 2))
    #
    
    '''

    pg.display.flip()

    previousTime = currentTime
    currentTime = pg.time.get_ticks()
    deltaTime = currentTime - previousTime


pg.quit()

