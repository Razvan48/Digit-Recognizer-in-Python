import pygame as pg
import numpy as np
import cv2 as cv


import Model





SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

FPS = 60

WIDTH_PERCENT_DRAWING_AREA = 0.7
CATEGORY_OFFSET_PERCENT = 0.05
SCORE_OFFSET_PERCENT = -0.025

NUM_DRAWING_PIXELS_ON_WIDTH = 28
NUM_DRAWING_PIXELS_ON_HEIGHT = 28

PIXEL_WIDTH = SCREEN_WIDTH * WIDTH_PERCENT_DRAWING_AREA / NUM_DRAWING_PIXELS_ON_WIDTH
PIXEL_HEIGHT = SCREEN_HEIGHT / NUM_DRAWING_PIXELS_ON_HEIGHT

BRUSH_SIZE_DRAWING = 1
BRUSH_SIZE_ERASING = 3

PREDICTION_HEIGHT = SCREEN_HEIGHT / len(Model.CATEGORIES)

drawingBoard = [[(0, 0, 0) for _ in range(NUM_DRAWING_PIXELS_ON_WIDTH)] for _ in range(NUM_DRAWING_PIXELS_ON_HEIGHT)]
drawingSpeed = 4

prediction = None

def draw():
    pg.draw.rect(screen, (50, 50, 50), (SCREEN_WIDTH * WIDTH_PERCENT_DRAWING_AREA, 0, SCREEN_WIDTH * (1 - WIDTH_PERCENT_DRAWING_AREA), SCREEN_HEIGHT))

    for i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
        for j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
            pg.draw.rect(screen, drawingBoard[i][j], (j * PIXEL_WIDTH, i * PIXEL_HEIGHT, PIXEL_WIDTH, PIXEL_HEIGHT))

    for category in Model.CATEGORIES:
        text = font.render(str(category), True, (255, 255, 255))
        screen.blit(text, (SCREEN_WIDTH * (WIDTH_PERCENT_DRAWING_AREA + CATEGORY_OFFSET_PERCENT), category * PREDICTION_HEIGHT))

    if prediction is not None:
        for category in Model.CATEGORIES:
            percentage = str(round(prediction[0][category] * 100, 2)) + '%'

            color = (255, 255, 255)
            if category == np.argmax(prediction):
                color = (0, 255, 0)

            text = font.render(percentage, True, color)
            screen.blit(text, (SCREEN_WIDTH * (WIDTH_PERCENT_DRAWING_AREA + (1.0 - WIDTH_PERCENT_DRAWING_AREA) / 2 +
                                               SCORE_OFFSET_PERCENT), category * PREDICTION_HEIGHT))




def dist(i1, j1, i2, j2):
    return max(abs(i1 - i2), abs(j1 - j2))


def update():
    x, y = pg.mouse.get_pos()
    if x < SCREEN_WIDTH * WIDTH_PERCENT_DRAWING_AREA:
        i = int(y / PIXEL_HEIGHT)
        j = int(x / PIXEL_WIDTH)
        if pg.mouse.get_pressed()[0]:
            for crtI in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
                for crtJ in range(NUM_DRAWING_PIXELS_ON_WIDTH):
                    if dist(i, j, crtI, crtJ) > BRUSH_SIZE_DRAWING:
                        continue
                    drawingBoard[crtI][crtJ] = (drawingBoard[crtI][crtJ][0] + drawingSpeed * deltaTime,
                                                drawingBoard[crtI][crtJ][1] + drawingSpeed * deltaTime,
                                                drawingBoard[crtI][crtJ][2] + drawingSpeed * deltaTime)
                    drawingBoard[crtI][crtJ] = (min(255, drawingBoard[crtI][crtJ][0]),
                                                min(255, drawingBoard[crtI][crtJ][1]),
                                                min(255, drawingBoard[crtI][crtJ][2]))
        elif pg.mouse.get_pressed()[2]:
            for crtI in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
                for crtJ in range(NUM_DRAWING_PIXELS_ON_WIDTH):
                    if dist(i, j, crtI, crtJ) > BRUSH_SIZE_ERASING:
                        continue
                    drawingBoard[crtI][crtJ] = (drawingBoard[crtI][crtJ][0] - drawingSpeed * deltaTime,
                                                drawingBoard[crtI][crtJ][1] - drawingSpeed * deltaTime,
                                                drawingBoard[crtI][crtJ][2] - drawingSpeed * deltaTime)
                    drawingBoard[crtI][crtJ] = (max(0, drawingBoard[crtI][crtJ][0]),
                                                max(0, drawingBoard[crtI][crtJ][1]),
                                                max(0, drawingBoard[crtI][crtJ][2]))



def predict():
    global prediction
    global drawingBoard

    drawingBoardInfo = np.zeros((NUM_DRAWING_PIXELS_ON_HEIGHT, NUM_DRAWING_PIXELS_ON_WIDTH))
    for i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
        for j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
            drawingBoardInfo[i][j] = drawingBoard[i][j][0] / 255.0

    minRow = NUM_DRAWING_PIXELS_ON_HEIGHT
    maxRow = -1
    minCol = NUM_DRAWING_PIXELS_ON_WIDTH
    maxCol = -1

    for i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
        for j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
            if drawingBoardInfo[i][j] > Model.GRAY_THRESHOLD:
                minRow = min(minRow, i)
                maxRow = max(maxRow, i)
                minCol = min(minCol, j)
                maxCol = max(maxCol, j)

    if minRow <= maxRow and minCol <= maxCol:
        drawingBoardInfo = drawingBoardInfo[minRow:maxRow + 1, minCol:maxCol + 1]

        drawingBoardInfo = cv.resize(drawingBoardInfo, (28, 28))
        drawingBoardInfo = drawingBoardInfo.reshape(28, 28, 1)

        prediction = Model.cnnModel.predict(drawingBoardInfo.reshape(-1, 28, 28, 1))
        print('Prediction:', np.argmax(prediction))





pg.init()

screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pg.display.set_caption('Digit Recognizer')

font = pg.font.SysFont('Arial', 36)

isRunning = True
clock = pg.time.Clock()
while isRunning:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            isRunning = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                isRunning = False
            elif event.key == pg.K_c or event.key == pg.K_C:
                drawingBoard = [[(0, 0, 0) for _ in range(NUM_DRAWING_PIXELS_ON_WIDTH)]
                                for _ in range(NUM_DRAWING_PIXELS_ON_HEIGHT)]
                prediction = None

    update()
    predict()
    screen.fill((0, 0, 0))
    draw()

    pg.display.flip()

    deltaTime = clock.tick(FPS)
