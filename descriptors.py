import numpy
import cv2  
from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo

import mahotas

def haralick(data): 
    data = cv2.imread(data, 0)
    return mahotas.features.haralick(data).mean(0)

def haralickconc(data):
    data = cv2.imread(data, 0)
    return mahotas.features.haralick(data).mean(0).tolist()

def glcm(data):
    data = cv2.imread(data, 0)
    glcm = graycomatrix(data, [2],[0], 1024, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0, 0]
    cont = graycoprops(glcm, 'contrast')[0, 0]
    corr = graycoprops(glcm, 'correlation')[0, 0]
    ener = graycoprops(glcm, 'energy')[0, 0]
    homo = graycoprops(glcm, 'homogeneity')[0, 0]
    return [diss, cont, corr,ener, homo]


def Bitdesc(data):
    data = cv2.imread(data, 0)
    return bio_taxo(data)




def haralick_glcm(data):
    return haralickconc(data) + glcm(data)

def haralick_bit(data):
    return haralickconc(data) + Bitdesc(data)
