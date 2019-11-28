'''
Seguindo passos mencionados no artigo Morphology-Based Hierarchical Representation with
Application to Text Segmentation in Natural Images.
'''

import cv2 as cv
import numpy as np

def main():

    A = cv.imread('img/img1.png')
    # Passo 1:
    img_cinza = cv.cvtColor(A,cv.COLOR_BGR2GRAY)
    kernel_size = 3

    # setando parâmetros do elemento estruturante
    morph_elem = cv.MORPH_RECT
    morph_size = 11
    element = cv.getStructuringElement(morph_elem, (morph_size, morph_size), (-1,-1))

    # Passo 2: Computando dilatação
    img_dilatacao = cv.morphologyEx(img_cinza, cv.MORPH_DILATE, element)
    cv.imshow('Imagem apos dilatacao', img_dilatacao)
    
    # Passo 2: Computando erosão
    img_erosao = cv.morphologyEx(img_cinza, cv.MORPH_ERODE, element)
    cv.imshow('Imagem apos erosao', img_erosao)

    # Passo 2: Computando gradiente morfológico
    img_grdmorf = img_dilatacao - img_erosao
    cv.imshow('Gradiente morfologico: dilatacao - erosao', img_grdmorf)

    # Passo 2: Aplicando Laplace function -> definida no artigo
    lapla = img_dilatacao + img_erosao + 2*img_cinza
    cv.imshow('Laplace', lapla)

    # Passo 3: Computando imagem interpolada
    interpol_lapla = cv.resize(lapla, None, fx = 4, fy = 4, interpolation = cv.INTER_NEAREST)
    cv.imshow('4x maior', interpol_lapla)

    cv.imshow('Imagem niveis de cinza', img_cinza)
    
    
    cv.waitKey(0)
    cv.destroyAllWindows()


main()