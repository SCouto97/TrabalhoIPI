'''
Seguindo passos mencionados no artigo Morphology-Based Hierarchical Representation with
Application to Text Segmentation in Natural Images.
'''

import cv2 as cv
import numpy as np

blocks = set()

def is_homogeneous(x,y):
    return (x,y) in blocks

def norm8(I):
    mn = I.min()
    mx = I.max()

    mx -= mn

    if mx == 0:
        return np.zeros((I.shape), np.uint8)

    I = (I - mn)/mx
    I *= 255
    return I.astype(np.uint8)

def main():

    div = 2.0
    k = 2
    threshold = 3

    A = cv.imread('img/teste4.png')
    A = cv.resize(A, ( int(A.shape[1]/div),int(A.shape[0]/div)))

    img_cinza = cv.cvtColor(A,cv.COLOR_BGR2GRAY)

    height, width = img_cinza.shape


    W = []

    w1 = np.zeros((k, k))

    for i in range(0,k):
        for j in range(0,k):
            if i%2 == 0:
                if j%2 == 0:
                    w1[i][j] = 1
                else:
                    w1[i][j] = -1
            else:
                if j%2 == 0:
                    w1[i][j] = -1
                else:
                    w1[i][j] = 1

    w2 = np.zeros((k,k))

    for i in range(0,k):
        for j in range(0,k):
            if j%2 == 0:
                w2[i][j] = 1
            else:
                w2[i][j] = -1


    w3 = np.zeros((k,k))

    for i in range(0,k):
        for j in range(0,k):
            if i%2 == 0:
                w3[i][j] = 1
            else:
                w3[i][j] = -1


    W.append(w1)
    W.append(w2)
    W.append(w3)

    # print(W);

    delta = np.zeros(3)

    for i in range(0, height):
        for j in range(0, width):
            if((i % k == 0) and (j % k == 0)):
                for m in range(0, k):
                    for n in range(0,k):
                        if(i + m < height and j + n < width):
                            for p in range(len(W)):
                                delta[p] += img_cinza[i+m][j+n] * W[p][m][n]
                delta *= 2
                delta /= k**2
                norm_val = cv.norm(delta, cv.NORM_INF)
                if norm_val < threshold:
                    blocks.add((i,j))

    img_final = img_cinza.copy()

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

    homogeneous_img = np.zeros((height, width), np.float32)

    for b in homo_blocks:
        i = b[0]
        j = b[1]

        if((i % k == 0) and (j % k == 0)):
            for m in range(0, k):
                for n in range(0,k):
                    if(i + m < height and j + n < width):
                        img_final[i+m][j+n] = 0
                        homogeneous_img[i+m][j+n] = img_cinza[i+m][j+n]/255

    for i in range(0, height):
        for j in range(0, width):
            if((i % k == 0) and (j % k == 0)):
                if(i,j) not in homo_blocks:
                    for m in range(0, k):
                        for n in range(0,k):
                            if(i + m < height and j + n < width):
                                homogeneous_img[i+m][j+n] = -1000

    im_floodfill = homogeneous_img.copy()

    while(1):
        cv.imshow('Imagem com blocos',homogeneous_img)
        k = cv.waitKey(0)
        if k == 27:
            exit(0)
        else:
            break

    visited = set()
    r = 5.0

    lo = 5/255.0
    hi = 5/255.0

    for i in range(0, height):
        for j in range(0, width):    
            if (i,j) in homo_blocks:
                if im_floodfill[i][j] not in visited:
                    prev = im_floodfill.copy()
                    rect = cv.floodFill(im_floodfill, None, (j, i), r, loDiff=lo, upDiff=hi)
                    visited.add(r) 
                    r += 10

                    diff = rect[1] - prev
                    im_uint = norm8(diff)

                    x0 = 10000
                    y0 = 10000

                    x1 = 0
                    y1 = 0
                    count = 0
                    for a in range(0,height):
                        for b in range(0, width):
                            if im_uint[a][b] != 0:
                                count += 1
                                x0 = min(x0,a)
                                y0 = min(y0,b)
                                x1 = max(x1,a)
                                y1 = max(y1,b)
                    
                    im_final = im_uint[x0:x1,y0:y1]

                    row_padd_init = 1
                    col_padd_init = 1

                    rows = im_final.shape[0]
                    cols = im_final.shape[1]
                    padding_rows = im_final.shape[0]+2
                    padding_cols = im_final.shape[1]+2
                    im_padding = np.zeros((padding_rows, padding_cols), np.uint8)

                    # copiando e fazendo shift da imagem no padding
                    for m in range(row_padd_init, (row_padd_init+im_final.shape[0])):
                        for n in range(col_padd_init, (col_padd_init+im_final.shape[1])):
                            im_padding[m,n] = im_final[m-row_padd_init, n-col_padd_init]
                    
                    # cv.imshow("rect", im_padding)
                    # cv.waitKey(0)   

                    white = np.sum(im_padding > 200)
                    cv.floodFill(im_padding, None, (0, 0), 255)

                    components = 0
                    for m in range(0, rows):
                        for n in range(0, cols):
                            if im_padding[m][n] < 10:
                                ret = cv.floodFill(im_padding, None, (n, m), 255)
                                if ret[0] > (white / 60):
                                    components += 1

                    if components > 0:
                        # contours, hierarchy = cv.findContours(im_uint, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        contours, hierarchy = cv.findContours(im_uint, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                        hull_list = []
                        for index in range(len(contours)):
                            hull_list.append(cv.convexHull(contours[index]))
                        cv.drawContours(A, hull_list, -1, (0, 255, 0), 3) 
                        # cv.rectangle(A,(y0,x0),(y1,x1),(0,255,0),2)

                    print("rect:", x0 , y0, x1, y1)
                    print("components: ", components)


    print(len(visited))
    cv.imshow("final", A)
    cv.waitKey(0)

#    cv.imshow('Painted blocks',img_final)
#    cv.waitKey(0)   
    cv.destroyAllWindows() 
"""

    cv.imshow('Floodfill',im_floodfill)
    cv.waitKey(0)
"""


main()