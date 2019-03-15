"""
@file    ImageUtils.py
@author  rohithjayarajan
@date 02/25/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
from scipy import interpolate


class ImageUtils:

    def ShowImage(self, Image, ImgName):
        cv2.imshow(ImgName, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def dlibFODObjectToNumpy(self, FODObject):
        points = np.zeros((68, 2), dtype="int")

        for idX in range(68):
            points[idX] = (FODObject.part(idX).x, FODObject.part(idX).y)

        return points

    def ConvertTo3D(self, Vector2D):
        Vector3D = np.array([Vector2D[1], Vector2D[0], 1])
        return Vector3D

    def ConvertToHomogeneous(self, Point):
        return Point/float(Point[2])

    def getTriangleBBox(self, triangle):
        xMin = min(triangle[0], triangle[2], triangle[4])
        yMin = min(triangle[1], triangle[3], triangle[5])
        xMax = max(triangle[0], triangle[2], triangle[4])
        yMax = max(triangle[1], triangle[3], triangle[5])

        return xMin, yMin, xMax, yMax

    def FuncUofR(self, r):
        Uval = -r**2 * (np.log10(r**2))
        return np.nan_to_num(Uval)

    def getRGBforInterp(self, SrcImage, x, y, gridSize=3):
        xPoints = np.arange(x-gridSize, x+gridSize+1, 1)
        yPoints = np.arange(y-gridSize, y+gridSize+1, 1)
        xI = []
        yI = []
        R = []
        G = []
        B = []
        for x in xPoints:
            for y in yPoints:
                xI.append(x)
                yI.append(y)
                B.append(SrcImage[x][y][0])
                G.append(SrcImage[x][y][1])
                R.append(SrcImage[x][y][2])
        xI = np.array(xI)
        yI = np.array(yI)
        B = np.array(B)
        G = np.array(G)
        R = np.array(R)
        fB = interpolate.interp2d(xI, yI, B, kind='linear')
        fG = interpolate.interp2d(xI, yI, G, kind='linear')
        fR = interpolate.interp2d(xI, yI, R, kind='linear')

        interpB = fB(x, y)
        interpG = fG(x, y)
        interpR = fR(x, y)
        return interpB, interpG, interpR
