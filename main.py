import cv2 as cv
import numpy as np

def main():

    A = cv.imread('img/img1.png')
    A_gs = cv.cvtColor(A,cv.COLOR_BGR2GRAY)

    cv.imshow('Final',A)
    cv.waitKey(0)

    cv.destroyAllWindows()


main()