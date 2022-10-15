import pygame
import sys
from pygame import image
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2


WINDOWSIZEX: int = 640
WINDOWSIZEY: int = 480
BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False
PREDICT = True
# Load our model
MODEL = load_model('handwritten.model')
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize our pygame
pygame.init()
FONT = pygame.font.SysFont("Arial", 10)
DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digits Recognition Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

# Keep on running until window is closed
while True:
    # When user write something, predict it
    for event in pygame.event.get():
        # if window is closed the quit the script
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # When user is writting record it
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        # When user press button start the writing
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        # when user stops then take what is written from the screen
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            # get what is written recently
            rect_min_x, rect_max_x = max(
                number_xcord[0]-BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYINC)
            rect_min_y, rect_max_y = max(
                number_ycord[0]-BOUNDARYINC, 0), min(WINDOWSIZEY, number_ycord[-1]+BOUNDARYINC)

            number_xcord = []
            number_ycord = []
            # store the data in img_arr
            img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[
                rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            # save the image if IMAGESAVE is true
            if IMAGESAVE:
                cv2.imwrite("image{image_cnt}.png", img_arr)
                image_cnt += 1

            # start to predict
            if PREDICT:
                # resize the image
                image = cv2.resize(img_arr, (28, 28),
                                   interpolation=cv2.INTER_AREA)
                image = np.pad(image, (5, 5), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))/255

                # predict the image and store the prediction in label
                label = str(
                    LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28)))])

                # make a text box to update the label
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURFACE.blit(textSurface, textRecObj)
            # if esc is pressed, clean the screen
            if event.type == KEYDOWN:
                if event.unicode == "U+001B":
                    DISPLAYSURFACE.fill(BLACK)

        pygame.display.update()
