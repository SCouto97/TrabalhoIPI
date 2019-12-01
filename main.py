'''
Seguindo passos mencionados no artigo Morphology-Based Hierarchical Representation with
Application to Text Segmentation in Natural Images.
'''

import cv2 as cv
import numpy as np
from queue import *

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

def bfsFill(img, point, offset, img_cinza, avg, dev):
    black_threshold = 10
    fill_color = 255
    height, width = img.shape

    if (img[point[0]][point[1]] > black_threshold):
        return 0, 0

    roff = [-1,0,1,0]
    coff = [0,1,0,-1]

    q = Queue()
    q.put(point)
    total = 0
    color_sum = 0
    total_text = 0
    while not q.empty():
        cur = q.get()
        x = cur[0] - offset[0]
        y = cur[1] - offset[1]
        if abs(img_cinza[x][y] - avg) > 2*dev:
            total_text += 1
            color_sum += img_cinza[x][y]
       
        total += 1

        for k in range(0,len(roff)):
            i = cur[0] + roff[k]
            j = cur[1] + coff[k]

            if (i >= 0 and i < height and j >= 0 and j < width and img[i][j] <= black_threshold):
                img[i][j] = fill_color 
                q.put((i,j))

    return total, total_text, color_sum

def detect_text(A, div, k, threshold, thresh_color):

    A = cv.resize(A, ( int(A.shape[1]/div),int(A.shape[0]/div)))
    has_text = False

    img_cinza = cv.cvtColor(A,cv.COLOR_BGR2GRAY)

#    kernel = np.ones((5,5),np.float32)/25
#    img_cinza = cv.filter2D(img_cinza,-1,kernel)

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
                  
                    im_aux = im_uint.copy()
                    im_uint = im_uint[x0:x1,y0:y1]
                    rows, cols = im_uint.shape
                    
                    if(cols == 0 or rows == 0):
                        continue

                    # print("ros:", rows, " c ",  cols)
                    # cv.imshow("final", im_uint)
                    # cv.waitKey(0)

                    for m in range(0, rows):
                        cv.floodFill(im_uint, None, (0, m), 255)
                    for m in range(0, cols):
                        cv.floodFill(im_uint, None, (m, 0), 255)
                    for m in range(rows):
                        cv.floodFill(im_uint, None, (cols-1, m), 255)
                    for m in range(cols):
                        cv.floodFill(im_uint, None, (m, rows-1), 255)

                    C = np.multiply((im_aux > 200), img_cinza)
                    total_bg = np.sum(im_uint > 200)
                    dev = np.std(C)
                    color_sum_bg = np.sum(C)
                    avg = color_sum_bg/total_bg

                    components = 0
                    for m in range(0, rows):
                        for n in range(0, cols):
                            if im_uint[m][n] < 10:
                                total, total_text, color_sum = bfsFill(im_uint, (m,n), (x0,y0), img_cinza, avg, dev)
                                if total > (total_bg / 60):
                                    if total_text != 0:
                                        print("valor: ",abs(color_sum/total_text - color_sum_bg/total_bg))

                                    if (total_text != 0) and (abs(color_sum/total_text - color_sum_bg/total_bg) >= thresh_color):
                                        components += 1
                    
                    if components > 0:
                        has_text = True
                        contours, hierarchy = cv.findContours(im_aux, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                        hull_list = []
                        for index in range(len(contours)):
                            hull_list.append(cv.convexHull(contours[index]))
                        cv.drawContours(A, hull_list, -1, (0, 255, 0), 2) 

#                    print("rect:", x0 , y0, x1, y1)
                    print("components: ", components)


 #   print(len(visited))
    return has_text, A

def main():

    div = 5.0
    k = 2
    thresh = 30
    thresh_color = 60

    # has_text, img = detect_text(cv.imread('img/img51.jpg'), div, k, thresh, thresh_color)
    has_text, img = detect_text(cv.imread('dataset/img51.jpg'), div, k, thresh, thresh_color)
    
    if has_text:
        print("possui texto na imagem")
    else:
        print("imagem nao possui texto")

    cv.imshow("final", img)
    cv.waitKey(0)

    cv.destroyAllWindows() 

main()