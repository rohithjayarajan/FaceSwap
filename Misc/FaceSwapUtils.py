"""
@file    FaceSwapUtils.py
@author  rohithjayarajan
@date 02/25/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
import argparse
import math
import dlib
from Misc.ImageUtils import ImageUtils

debug = False


class FaceSwapUtils:
    def __init__(self, ModelPath):
        self.HelperFunctions = ImageUtils()
        self.FaceDetector = dlib.get_frontal_face_detector()
        self.ShapePredictor = dlib.shape_predictor(ModelPath)

    def getTriangulationCorrespondence(self, LandMarkPointsTarget, DelaunayTriangleList, LandMarkPointsSrc):
        DelaunayTriangleListSrc = []
        resultShape = DelaunayTriangleList.shape

        for triangle in DelaunayTriangleList:
            a = (triangle[0], triangle[1])
            idA = LandMarkPointsTarget.index(a)
            b = (triangle[2], triangle[3])
            idB = LandMarkPointsTarget.index(b)
            c = (triangle[4], triangle[5])
            idC = LandMarkPointsTarget.index(c)
            triangleIndices = LandMarkPointsSrc[idA], LandMarkPointsSrc[idB], LandMarkPointsSrc[idC]
            DelaunayTriangleListSrc.append(triangleIndices)
        DelaunayTriangleListSrc = np.array(DelaunayTriangleListSrc)
        DelaunayTriangleListSrc = np.reshape(
            DelaunayTriangleListSrc, resultShape)

        return DelaunayTriangleListSrc

    def getMatrixBary(self, DelaunayTriangleList):
        BMatrix = []
        BInvMatrix = []
        for triangle in DelaunayTriangleList:
            ax = triangle[0]
            ay = triangle[1]
            bx = triangle[2]
            by = triangle[3]
            cx = triangle[4]
            cy = triangle[5]
            B = np.array([[ax, bx, cx], [ay, by, cy], [1, 1, 1]])
            BMatrix.append(B)
            BInv = np.linalg.inv(B)
            BInvMatrix.append(BInv)
        BMatrix = np.array(BMatrix)
        BInvMatrix = np.array(BInvMatrix)

        return BMatrix, BInvMatrix

    def Blending(self, TargetImg, SrcImg, Mask):
        radius = 3
        kernel = np.ones((radius, radius), np.uint8)
        Mask = cv2.dilate(Mask, kernel, iterations=1)
        r = cv2.boundingRect(Mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        return cv2.seamlessClone(
            TargetImg, SrcImg, Mask, center, cv2.NORMAL_CLONE)

    def DetectFacialLandmarks(self, InputImage):
        GrayImage = cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY)
        dets = self.FaceDetector(GrayImage, 1)
        isFace = False
        for (_, det) in enumerate(dets):
            isFace = True
            shape = self.ShapePredictor(GrayImage, det)
            shape = self.HelperFunctions.dlibFODObjectToNumpy(shape)
            DelaunayBBox = (0, 0, InputImage.shape[1], InputImage.shape[0])
            LandMarkPoints = []

            for (x, y) in shape:
                LandMarkPoints.append((x, y))
                if(debug):
                    print("x: {} y: {}".format(x, y))
                    cv2.circle(InputImage, (x, y), 7, (0, 255, 0), -1)

            if(debug):
                print("Face Landmark Points: {}".format(LandMarkPoints))
        if(isFace):
            return isFace, DelaunayBBox, LandMarkPoints
        else:
            return isFace, [], []

    def DetectFacialLandmarks2(self, InputImage):
        GrayImage = cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY)
        dets = self.FaceDetector(GrayImage, 1)
        isFace = False
        LandMarkPoints = []
        for (idX, det) in enumerate(dets):
            isFace = True
            shape = self.ShapePredictor(GrayImage, det)
            shape = self.HelperFunctions.dlibFODObjectToNumpy(shape)
            DelaunayBBox = (0, 0, InputImage.shape[1], InputImage.shape[0])
            LandMarkPoints1 = []

            for (x, y) in shape:
                LandMarkPoints1.append((x, y))
                if(debug):
                    print("x: {} y: {}".format(x, y))
                    cv2.circle(InputImage, (x, y), 7, (0, 255, 0), -1)
            LandMarkPoints.append(LandMarkPoints1)
            if(idX == 1):
                break
            if(debug):
                print("Face Landmark Points: {}".format(LandMarkPoints))
        if(isFace):
            return isFace, DelaunayBBox, LandMarkPoints
        else:
            return isFace, [0], [0]

    def GetDelaunayTriangulation(self, DelaunayBBox, LandMarkPoints, InputImage):
        subdiv = cv2.Subdiv2D(DelaunayBBox)
        subdiv.insert(LandMarkPoints)
        DelaunayTriangleList = subdiv.getTriangleList()
        DelaunayTriangleList = DelaunayTriangleList[DelaunayTriangleList.min(
            axis=1) >= 0, :]
        DelaunayTriangleList = DelaunayTriangleList[DelaunayTriangleList.max(
            axis=1) <= max(InputImage.shape[0], InputImage.shape[1]), :]
        DelaunayTriangleListN = []
        for triangle in DelaunayTriangleList:
            a = (triangle[0], triangle[1])
            b = (triangle[2], triangle[3])
            c = (triangle[4], triangle[5])
            if(self.HelperFunctions.isGoodAngle(a, b, c, 140)):
                DelaunayTriangleListN.append(triangle)
        DelaunayTriangleListN = np.array(DelaunayTriangleListN)

        if(debug):
            # print("DelaunayTriangleList: {}".format(DelaunayTriangleList))
            # print(type(DelaunayTriangleList))
            for triangle in DelaunayTriangleList:
                a = (triangle[0], triangle[1])
                b = (triangle[2], triangle[3])
                c = (triangle[4], triangle[5])
                cv2.line(InputImage, a, b, (0, 0, 255), 1, cv2.LINE_AA, 0)
                cv2.line(InputImage, b, c, (0, 0, 255), 1, cv2.LINE_AA, 0)
                cv2.line(InputImage, c, a, (0, 0, 255), 1, cv2.LINE_AA, 0)
            self.HelperFunctions.ShowImage(InputImage, 'Facial Landmarks')
        return DelaunayTriangleList
