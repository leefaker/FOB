# encoding:utf-8
import cv2
import numpy as np
import match
from sklearn.externals import joblib
from skimage.feature import greycomatrix, greycoprops

def GetGLCMFeature(img):
    scale = [2, 3, 4]
    theta = [0, np.pi / 4, np.pi / 2, np.pi / 4 * 3]
    featureType = ['dissimilarity', 'homogeneity', 'correlation', 'contrast']
    downratio = int(4)
    singletypesize = int(len(scale) * len(theta))
    lensize = int(singletypesize * len(featureType))
    result_array = np.zeros((1, lensize), dtype=np.float64)
    img = img / downratio
    img = img.astype(np.uint8)
    glcm_img = greycomatrix(img, scale, theta, int(256 / downratio), True, True)
    col = 0
    for featureName in featureType:
        feature = greycoprops(glcm_img, featureName)
        feature.shape = [1, feature.shape[0] * feature.shape[1]]
        result_array[0, col:col + singletypesize] = feature[:]
        col = col + singletypesize
    return result_array


def classify_fob(data,dataTransform,SVM):
    dataNew = np.array(data).reshape(1, -1)
    DataSt = dataTransform.transform(dataNew)
    label = SVM.predict(DataSt)
    return label
def predict(img):
    scale = [2, 3, 4]
    theta = [0, np.pi / 4, np.pi / 2, np.pi / 4 * 3]
    featureType = ['dissimilarity', 'homogeneity', 'correlation', 'contrast']
    singletypesize = int(len(scale) * len(theta))
    lensize = int(singletypesize * len(featureType))
    resizeHeight = 115
    resizeWidth = 200
    dataTransform = joblib.load("glcm_transform.m")
    SVM = joblib.load("glcm_train.m")
    glcm_array = np.zeros(lensize * 3, dtype=np.float64)
    img=match.get_ori(img)
    img=cv2.medianBlur(img,7)
    img = cv2.resize(img, (resizeHeight, resizeWidth), 0, 0, interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(img)
    feature_b = GetGLCMFeature(b)
    feature_g = GetGLCMFeature(g)
    feature_r = GetGLCMFeature(r)
    glcm_array[0:lensize] = feature_b[:]
    glcm_array[lensize:lensize * 2] = feature_g[:]
    glcm_array[lensize * 2:] = feature_r[:]
    predict_label = classify_fob(glcm_array, dataTransform, SVM)
    return predict_label











