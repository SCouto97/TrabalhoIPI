'''
Seguindo passos mencionados no artigo Morphology-Based Hierarchical Representation with
Application to Text Segmentation in Natural Images.
'''

import cv2 as cv
import numpy as np

blocks = set()

def is_homogeneous(x,y):
    return (x,y) in blocks

def main():

    A = cv.imread('img/img3.jpg')
    # Passo 1:
    img_cinza = cv.cvtColor(A,cv.COLOR_BGR2GRAY)

    width, height = img_cinza.shape

    cv.imshow('Grayscale',img_cinza)
    cv.waitKey(0)
   
    kernel = np.ones((5,5),np.float32)/25
    img_blur = cv.filter2D(img_cinza,-1,kernel)

    k = 4

    W = []
    threshold = 15.0

    w1 = np.zeros((k, k))

    for i in range(0,k):
        for j in range(0,k):
            if(j < k/2):
                if(i < k/2):
                    w1[i][j] = 1
                else:
                    w1[i][j] = -1
            else:
                if(i >= k/2):
                    w1[i][j] = 1
                else:
                    w1[i][j] = -1

    w2 = np.zeros((k,k))

    for i in range(0,k):
        for j in range(0,k):
            if(i < k/2):
                w2[i][j] = -1
            else:
                w2[i][j] = 1


    w3 = np.zeros((k,k))

    for i in range(0,k):
        for j in range(0,k):
            if(j < k/2):
                w1[i][j] = 1
            else:
                w1[i][j] = -1

    W.append(w1)
    W.append(w2)
    W.append(w3)

    delta = np.zeros(3)


    for i in range(0, height):
        for j in range(0, width):
            if((i % k == 0) and (j % k == 0)):
                for m in range(0, k):
                    for n in range(0,k):
                        if(i + m < height and j + n < width):
                            for p in range(0,2):
                                delta[p] += img_cinza[j+n][i+m] * W[p][m][n]
                #for i in range(0,2):
                #    print("delta = ", delta[i])
                delta *= 2
                delta /= k**2
                norm_val = cv.norm(delta, cv.NORM_INF)
                if norm_val < threshold:
                    blocks.add((i,j))

    print(len(blocks))

    img_final = img_cinza.copy()
    img_final2 = img_final.copy()

    homo_blocks = set()

    for b in blocks:
        i = b[0]
        j = b[1]

        if is_homogeneous(i-k, j):
            homo_blocks.add((i-k,j))

        if is_homogeneous(i+k, j):
            homo_blocks.add((i+k,j))

        if is_homogeneous(i, j-k):
            homo_blocks.add((i,j-k))

        if is_homogeneous(i, j+k):
            homo_blocks.add((i,j+k))

        if((i % k == 0) and (j % k == 0)):
            for m in range(0, k):
                for n in range(0,k):
                    if(i + m < height and j + n < width):
                        img_final[j+n][i+m] = 0

    for b in homo_blocks:
        i = b[0]
        j = b[1]

        if((i % k == 0) and (j % k == 0)):
            for m in range(0, k):
                for n in range(0,k):
                    if(i + m < height and j + n < width):
                        img_final2[j+n][i+m] = 0

    cv.imshow('Painted blocks',img_final2)
    cv.waitKey(0)
   
   
    cv.destroyAllWindows()



main()