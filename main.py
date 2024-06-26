import pygame as pg
import numpy as np
import sklearn as skl
import sklearn.datasets as datasets
import pandas as pd

# Model

trainSize = 10000

mnist = datasets.fetch_openml('mnist_784', version=1, cache=True)

trainImages = np.array(mnist.data.astype(np.uint8)).reshape((-1, 28 * 28))[:trainSize]
trainLabels = np.array(mnist.target.astype(np.uint8))[:trainSize]

randomPermutation = np.random.permutation(trainSize)

trainImages = trainImages[randomPermutation]
trainLabels = trainLabels[randomPermutation]

# KNN
K = 100


def predictKNN(testImage, metric='l2'):
    global K
    global trainImages
    global trainLabels

    if metric == 'l1':
        distances = np.sum(np.abs(trainImages - testImage), axis=1)
    elif metric == 'l2':
        distances = np.sqrt(np.sum((trainImages - testImage) ** 2, axis=1))
    indices = distances.argsort()
    bestKIndices = indices[:K]
    bestKLabels = trainLabels[bestKIndices]
    return np.bincount(bestKLabels)


# Interface, PyGame

pg.init()

pixelWidth = 25
pixelHeight = 25

numPixelsDrawWidth = 28
numPixelsDrawHeight = 28

drawWidth = numPixelsDrawWidth * pixelWidth
drawHeight = numPixelsDrawHeight * pixelHeight

numPixelsPredictionWidth = 28

predictionWidth = numPixelsPredictionWidth * pixelWidth

screenWidth = drawWidth + predictionWidth
screenHeight = drawHeight

fontSize = 40
font = pg.font.Font(None, fontSize)

screen = pg.display.set_mode((screenWidth, screenHeight))

# clear
screen.fill((0, 0, 0))
drawMatrix = [[0] * numPixelsDrawWidth for y in range(numPixelsDrawHeight)]
drawCells = [[pg.Rect(x * pixelWidth, y * pixelHeight, pixelWidth, pixelHeight) for x in range(numPixelsDrawWidth)] for y in range(numPixelsDrawHeight)]  # x, y, width, height

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
                        if lin < 0 or lin >= numPixelsDrawHeight or col < 0 or col >= numPixelsDrawWidth:
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
                        if lin < 0 or lin >= numPixelsDrawHeight or col < 0 or col >= numPixelsDrawWidth:
                            continue
                        dist = abs(drawMatrixLine - lin) + abs(drawMatrixColumn - col)
                        if dist == 0:  # or dist >= clearRadius:
                            continue
                        drawMatrix[lin][col] = max(0, drawMatrix[drawMatrixLine][drawMatrixColumn] - clearSpeed * deltaTime // dist)

    binCountsKNN = predictKNN(np.array(drawMatrix).reshape((-1,)))

    clearPredictionScreen = pg.Rect(drawWidth, 0, predictionWidth, screenHeight)
    pg.draw.rect(screen, (50, 50, 50), clearPredictionScreen)

    for i in range(len(drawCells)):
        for j in range(len(drawCells[i])):
            luminosity = drawMatrix[i][j]
            pg.draw.rect(screen, (luminosity, luminosity, luminosity), drawCells[i][j])

    for label, frequency in enumerate(binCountsKNN):
        textToRender = str(label) + ': ' + str(round(100 * frequency / K, 2)) + '%'
        if label == np.argmax(binCountsKNN):
            textSurface = font.render(textToRender, True, (0, 255, 0))
        else:
            textSurface = font.render(textToRender, True, (255, 255, 255))
        screen.blit(textSurface, (drawWidth + predictionWidth // 2, label * screenHeight // 10 + fontSize // 2))

    pg.display.flip()

    previousTime = currentTime
    currentTime = pg.time.get_ticks()
    deltaTime = currentTime - previousTime


pg.quit()

