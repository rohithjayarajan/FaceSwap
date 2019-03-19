"""
@file    FaceSwapMethods.py
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
from Misc.FaceSwapUtils import FaceSwapUtils

debug = False

# Swap faces in two user defined images


class FaceSwapImages:
    def __init__(self, Image_Path_1, Image_Path_2, ModelPath):
        self.ModelPath = ModelPath
        InputImageList = [cv2.imread(Image_Path_1), cv2.imread(Image_Path_2)]
        self.Images = np.array(InputImageList)
        self.HelperFunctions = ImageUtils()
        self.FaceSwapFunctions = FaceSwapUtils(self.ModelPath)

    def FaceWarpByTriangulation(self, weight):
        TargetImg = self.Images[0]
        SrcImg = self.Images[1]
        CloneSrcImg = TargetImg.copy()
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)

        isFace1, DelaunayBBoxTarget, LandMarkPointsTarget = self.FaceSwapFunctions.DetectFacialLandmarks(
            TargetImg)
        if(not isFace1):
            return TargetImg
        DelaunayTrianglesTarget = self.FaceSwapFunctions.GetDelaunayTriangulation(
            DelaunayBBoxTarget, LandMarkPointsTarget, TargetImg)

        isFace2, _, LandMarkPointsSrc = self.FaceSwapFunctions.DetectFacialLandmarks(
            SrcImg)
        if(not isFace2):
            return TargetImg
        DelaunayTrianglesSrc = self.FaceSwapFunctions.getTriangulationCorrespondence(
            LandMarkPointsTarget, DelaunayTrianglesTarget, LandMarkPointsSrc)

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)

        _, BInvMatrix = self.FaceSwapFunctions.getMatrixBary(
            DelaunayTrianglesTarget)
        AMatrix, _ = self.FaceSwapFunctions.getMatrixBary(DelaunayTrianglesSrc)

        triangle_number = 0
        for triangle in DelaunayTrianglesTarget:
            xMin, yMin, xMax, yMax = self.HelperFunctions.getTriangleBBox(
                triangle)
            BInv = BInvMatrix[triangle_number]
            for x in range(int(xMin), int(xMax+1)):
                for y in range(int(yMin), int(yMax+1)):
                    BarycentricCoordinatesTarget = np.matmul(
                        BInv, np.array([[x], [y], [1]]))
                    if((BarycentricCoordinatesTarget[0] >= 0 and BarycentricCoordinatesTarget[0] <= 1) and (BarycentricCoordinatesTarget[1] >= 0 and BarycentricCoordinatesTarget[1] <= 1) and (BarycentricCoordinatesTarget[2] >= 0 and BarycentricCoordinatesTarget[2] <= 1)):
                        PixelLocSrc = np.matmul(
                            AMatrix[triangle_number], BarycentricCoordinatesTarget)
                        PixelLocSrc = np.divide(PixelLocSrc, PixelLocSrc[2])
                        TargetImg[y][x] = weight*SrcImg[int(
                            PixelLocSrc[1])][int(PixelLocSrc[0])] + (1-weight)*TargetImg[y][x]
            triangle_number += 1
        FaceSwapOP = self.FaceSwapFunctions.Blending(
            TargetImg, CloneSrcImg, Mask)
        if debug:
            self.HelperFunctions.ShowImage(TargetImg, 'Target Image')
            self.HelperFunctions.ShowImage(CloneSrcImg, 'Source Image')
            FaceSwapOP = cv2.resize(FaceSwapOP, (720, 720))
        # self.HelperFunctions.ShowImage(
        #     FaceSwapOP, 'FaceSwap using Delaunay Triangulation')
        return FaceSwapOP

    def FaceWarpByTPS(self):
        TargetImg = self.Images[0]
        SrcImg = self.Images[1]
        CloneSrcImg = TargetImg.copy()

        isFace1, _, LandMarkPointsTarget = self.FaceSwapFunctions.DetectFacialLandmarks(
            TargetImg)
        if(not isFace1):
            return TargetImg
        isFace2, _, LandMarkPointsSrc = self.FaceSwapFunctions.DetectFacialLandmarks(
            SrcImg)
        if(not isFace2):
            return TargetImg
        p = len(LandMarkPointsTarget)
        K = np.zeros([p, p])

        # K matrix
        for idX in range(p):
            for idY in range(p):
                K[idX][idY] = self.HelperFunctions.FuncUofR(
                    np.linalg.norm(np.array(LandMarkPointsTarget[idX]) - np.array(LandMarkPointsTarget[idY]), ord=2))

        # P matrix
        P = np.zeros([p, 3])
        for i, (x, y) in enumerate(LandMarkPointsTarget):
            P[i] = (x, y, 1)

        # lambda I matrix
        I = np.identity(p+3)
        lamb = 1e-4

        # V matrix
        vx = np.zeros([p+3])
        vy = np.zeros([p+3])

        for i, (x, y) in enumerate(LandMarkPointsSrc):
            vx[i] = x
            vy[i] = y
        vx = np.reshape(vx, (p+3, 1))
        vy = np.reshape(vy, (p+3, 1))

        t1 = np.hstack((K, P))
        t2 = np.hstack((np.transpose(P), np.zeros([3, 3])))
        M = np.vstack((t1, t2))

        resultX = np.dot(np.linalg.inv(M + lamb*I), vx)
        resultY = np.dot(np.linalg.inv(M + lamb*I), vy)

        wX = resultX[0:p]
        axX = resultX[p]
        axY = resultX[p+1]
        ax1 = resultX[p+2]

        wY = resultY[0:p]
        ayX = resultY[p]
        ayY = resultY[p+1]
        ay1 = resultY[p+2]

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)
        if debug:
            print("Points inside convex hull: {}".format(hull))
        Mask = np.zeros(
            (TargetImg.shape[0], TargetImg.shape[1], 3), np.uint8)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = Mask[:, :, 1]
        ptsY = np.where(Mask == 255)[0]
        ptsX = np.where(Mask == 255)[1]
        ptsY = np.transpose(ptsY)
        ptsX = np.transpose(ptsX)

        pts = np.vstack((ptsX, ptsY))
        pts = np.transpose(pts)

        if debug:
            print("K: {}".format(K))
            print("P: {}".format(P))
            print("M: {}".format(M))
            print("vx: {}".format(vx))
            print("vy: {}".format(vy))
            print("resultX: {}".format(resultX))
            print("resultY: {}".format(resultY))
            self.HelperFunctions.ShowImage(Mask, 'Convex hull for face')

        for Lpts in pts:
            U1 = (points - Lpts)
            U1 = np.linalg.norm(U1, ord=2, axis=1)
            U1 = self.HelperFunctions.FuncUofR(U1)
            wUX = np.matmul(np.transpose(wX), U1)
            wUY = np.matmul(np.transpose(wY), U1)
            fX = int(ax1 + axX*Lpts[0] + axY*Lpts[1] + wUX)
            fY = int(ay1 + ayX*Lpts[0] + ayY*Lpts[1] + wUY)
            TargetImg[Lpts[1]][Lpts[0]] = SrcImg[fY][fX]

        FaceSwapOP = self.FaceSwapFunctions.Blending(
            TargetImg, CloneSrcImg, Mask)
        # self.HelperFunctions.ShowImage(
        #     FaceSwapOP, 'FaceSwap using Thin Plate Spline')
        return FaceSwapOP


# Swap a face in video with user defined image

class FaceSwapVideoWithImage:
    def __init__(self, Image_Path, ModelPath):
        self.ModelPath = ModelPath
        # ImageTemp = cv2.resize(ImageTemp, (1000, 800))
        self.Image = cv2.imread(Image_Path)
        self.HelperFunctions = ImageUtils()
        self.FaceSwapFunctions = FaceSwapUtils(self.ModelPath)

    def FaceWarpByTriangulation(self, frame, weight):
        TargetImg = frame
        SrcImg = self.Image
        CloneSrcImg = TargetImg.copy()
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)

        isFace1, DelaunayBBoxTarget, LandMarkPointsTarget = self.FaceSwapFunctions.DetectFacialLandmarks(
            TargetImg)
        if(not isFace1):
            return TargetImg
        DelaunayTrianglesTarget = self.FaceSwapFunctions.GetDelaunayTriangulation(
            DelaunayBBoxTarget, LandMarkPointsTarget, TargetImg)

        isFace2, _, LandMarkPointsSrc = self.FaceSwapFunctions.DetectFacialLandmarks(
            SrcImg)
        if(not isFace2):
            return TargetImg
        DelaunayTrianglesSrc = self.FaceSwapFunctions.getTriangulationCorrespondence(
            LandMarkPointsTarget, DelaunayTrianglesTarget, LandMarkPointsSrc)

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)

        _, BInvMatrix = self.FaceSwapFunctions.getMatrixBary(
            DelaunayTrianglesTarget)
        AMatrix, _ = self.FaceSwapFunctions.getMatrixBary(DelaunayTrianglesSrc)

        triangle_number = 0
        for triangle in DelaunayTrianglesTarget:
            xMin, yMin, xMax, yMax = self.HelperFunctions.getTriangleBBox(
                triangle)
            BInv = BInvMatrix[triangle_number]
            for x in range(int(xMin), int(xMax+1)):
                for y in range(int(yMin), int(yMax+1)):
                    BarycentricCoordinatesTarget = np.matmul(
                        BInv, np.array([[x], [y], [1]]))
                    if((BarycentricCoordinatesTarget[0] >= 0 and BarycentricCoordinatesTarget[0] <= 1) and (BarycentricCoordinatesTarget[1] >= 0 and BarycentricCoordinatesTarget[1] <= 1) and (BarycentricCoordinatesTarget[2] >= 0 and BarycentricCoordinatesTarget[2] <= 1)):
                        PixelLocSrc = np.matmul(
                            AMatrix[triangle_number], BarycentricCoordinatesTarget)
                        PixelLocSrc = np.divide(PixelLocSrc, PixelLocSrc[2])
                        TargetImg[y][x] = weight*SrcImg[int(
                            PixelLocSrc[1])][int(PixelLocSrc[0])] + (1-weight)*TargetImg[y][x]
            triangle_number += 1
        FaceSwapOP = self.FaceSwapFunctions.Blending(
            TargetImg, CloneSrcImg, Mask)
        if debug:
            self.HelperFunctions.ShowImage(TargetImg, 'Target Image')
            self.HelperFunctions.ShowImage(CloneSrcImg, 'Source Image')
            FaceSwapOP = cv2.resize(FaceSwapOP, (720, 720))
        # self.HelperFunctions.ShowImage(
        #     FaceSwapOP, 'FaceSwap using Delaunay Triangulation')
        return FaceSwapOP

    def FaceWarpByTPS(self, frame):
        TargetImg = frame
        SrcImg = self.Image
        CloneSrcImg = TargetImg.copy()

        isFace1, _, LandMarkPointsTarget = self.FaceSwapFunctions.DetectFacialLandmarks(
            TargetImg)
        if(not isFace1):
            return TargetImg
        isFace2, _, LandMarkPointsSrc = self.FaceSwapFunctions.DetectFacialLandmarks(
            SrcImg)
        if(not isFace2):
            return TargetImg
        p = len(LandMarkPointsTarget)
        K = np.zeros([p, p])

        # K matrix
        for idX in range(p):
            for idY in range(p):
                K[idX][idY] = self.HelperFunctions.FuncUofR(
                    np.linalg.norm(np.array(LandMarkPointsTarget[idX]) - np.array(LandMarkPointsTarget[idY]), ord=2))

        # P matrix
        P = np.zeros([p, 3])
        for i, (x, y) in enumerate(LandMarkPointsTarget):
            P[i] = (x, y, 1)

        # lambda I matrix
        I = np.identity(p+3)
        lamb = 1e-4

        # V matrix
        vx = np.zeros([p+3])
        vy = np.zeros([p+3])

        for i, (x, y) in enumerate(LandMarkPointsSrc):
            vx[i] = x
            vy[i] = y
        vx = np.reshape(vx, (p+3, 1))
        vy = np.reshape(vy, (p+3, 1))

        t1 = np.hstack((K, P))
        t2 = np.hstack((np.transpose(P), np.zeros([3, 3])))
        M = np.vstack((t1, t2))

        resultX = np.dot(np.linalg.inv(M + lamb*I), vx)
        resultY = np.dot(np.linalg.inv(M + lamb*I), vy)

        wX = resultX[0:p]
        axX = resultX[p]
        axY = resultX[p+1]
        ax1 = resultX[p+2]

        wY = resultY[0:p]
        ayX = resultY[p]
        ayY = resultY[p+1]
        ay1 = resultY[p+2]

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)

        Mask = np.zeros(
            (TargetImg.shape[0], TargetImg.shape[1], 3), np.uint8)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = Mask[:, :, 1]
        ptsY = np.where(Mask == 255)[0]
        ptsX = np.where(Mask == 255)[1]
        ptsY = np.transpose(ptsY)
        ptsX = np.transpose(ptsX)

        pts = np.vstack((ptsX, ptsY))
        pts = np.transpose(pts)

        if debug:
            print("K: {}".format(K))
            print("P: {}".format(P))
            print("M: {}".format(M))
            print("vx: {}".format(vx))
            print("vy: {}".format(vy))
            print("resultX: {}".format(resultX))
            print("resultY: {}".format(resultY))
        # self.HelperFunctions.ShowImage(Mask, 'Convex hull for face')

        for Lpts in pts:
            U1 = (points - Lpts)
            U1 = np.linalg.norm(U1, ord=2, axis=1)
            U1 = self.HelperFunctions.FuncUofR(U1)
            wUX = np.matmul(np.transpose(wX), U1)
            wUY = np.matmul(np.transpose(wY), U1)
            fX = int(ax1 + axX*Lpts[0] + axY*Lpts[1] + wUX)
            fY = int(ay1 + ayX*Lpts[0] + ayY*Lpts[1] + wUY)
            if fX < 0 or fY < 0 or fY >= SrcImg.shape[0] or fX >= SrcImg.shape[1]:
                TargetImg[Lpts[1]][Lpts[0]] = TargetImg[Lpts[1]][Lpts[0]]
            else:
                TargetImg[Lpts[1]][Lpts[0]] = SrcImg[fY][fX]

        FaceSwapOP = self.FaceSwapFunctions.Blending(
            TargetImg, CloneSrcImg, Mask)
        # self.HelperFunctions.ShowImage(
        #     FaceSwapOP, 'FaceSwap using Thin Plate Spline')
        return FaceSwapOP

# Swap faces in video


class FaceSwapVideoFaces:
    def __init__(self, ModelPath):
        self.ModelPath = ModelPath
        self.HelperFunctions = ImageUtils()
        self.FaceSwapFunctions = FaceSwapUtils(self.ModelPath)

    def FaceWarpByTriangulation(self, frame, weight):
        TargetImg = frame.copy()
        SrcImg = frame.copy()
        CloneSrcImg = TargetImg.copy()
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)

        isFace1, DelaunayBBoxTarget, LandMarkPoints = self.FaceSwapFunctions.DetectFacialLandmarks2(
            TargetImg)
        if(not isFace1 or len(LandMarkPoints) < 2):
            return TargetImg
        LandMarkPointsTarget = LandMarkPoints[0]
        LandMarkPointsSrc = LandMarkPoints[1]

        DelaunayTrianglesTarget = self.FaceSwapFunctions.GetDelaunayTriangulation(
            DelaunayBBoxTarget, LandMarkPointsTarget, TargetImg)
        DelaunayTrianglesTarget2 = self.FaceSwapFunctions.GetDelaunayTriangulation(
            DelaunayBBoxTarget, LandMarkPointsSrc, TargetImg)

        DelaunayTrianglesSrc = self.FaceSwapFunctions.getTriangulationCorrespondence(
            LandMarkPointsTarget, DelaunayTrianglesTarget, LandMarkPointsSrc)

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
        # self.HelperFunctions.ShowImage(Mask, 'Mask')

        _, BInvMatrix = self.FaceSwapFunctions.getMatrixBary(
            DelaunayTrianglesTarget)
        AMatrix, _ = self.FaceSwapFunctions.getMatrixBary(
            DelaunayTrianglesSrc)

        triangle_number = 0
        for triangle in DelaunayTrianglesTarget:
            xMin, yMin, xMax, yMax = self.HelperFunctions.getTriangleBBox(
                triangle)
            BInv = BInvMatrix[triangle_number]
            for x in range(int(xMin), int(xMax+1)):
                for y in range(int(yMin), int(yMax+1)):
                    BarycentricCoordinatesTarget = np.matmul(
                        BInv, np.array([[x], [y], [1]]))
                    if((BarycentricCoordinatesTarget[0] >= 0 and BarycentricCoordinatesTarget[0] <= 1) and (BarycentricCoordinatesTarget[1] >= 0 and BarycentricCoordinatesTarget[1] <= 1) and (BarycentricCoordinatesTarget[2] >= 0 and BarycentricCoordinatesTarget[2] <= 1)):
                        PixelLocSrc = np.matmul(
                            AMatrix[triangle_number], BarycentricCoordinatesTarget)
                        PixelLocSrc = np.divide(PixelLocSrc, PixelLocSrc[2])
                        TargetImg[y][x] = weight*SrcImg[int(
                            PixelLocSrc[1])][int(PixelLocSrc[0])] + (1-weight)*TargetImg[y][x]
            triangle_number += 1
        TargetImg1 = self.FaceSwapFunctions.Blending(
            TargetImg, CloneSrcImg, Mask)
        CloneSrcImg = TargetImg1
        # self.HelperFunctions.ShowImage(TargetImg1, 'TargetImg1')
        #################################

        Mask = np.zeros(TargetImg1.shape, TargetImg1.dtype)

        DelaunayTrianglesSrc = self.FaceSwapFunctions.getTriangulationCorrespondence(
            LandMarkPointsSrc, DelaunayTrianglesTarget2, LandMarkPointsTarget)

        points = np.array(LandMarkPointsSrc)
        hull = cv2.convexHull(points)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
        # self.HelperFunctions.ShowImage(Mask, 'Mask')

        _, BInvMatrix = self.FaceSwapFunctions.getMatrixBary(
            DelaunayTrianglesTarget2)
        AMatrix, _ = self.FaceSwapFunctions.getMatrixBary(
            DelaunayTrianglesSrc)

        triangle_number = 0
        for triangle in DelaunayTrianglesTarget2:
            xMin, yMin, xMax, yMax = self.HelperFunctions.getTriangleBBox(
                triangle)
            BInv = BInvMatrix[triangle_number]
            for x in range(int(xMin), int(xMax+1)):
                for y in range(int(yMin), int(yMax+1)):
                    BarycentricCoordinatesTarget = np.matmul(
                        BInv, np.array([[x], [y], [1]]))
                    if((BarycentricCoordinatesTarget[0] >= 0 and BarycentricCoordinatesTarget[0] <= 1) and (BarycentricCoordinatesTarget[1] >= 0 and BarycentricCoordinatesTarget[1] <= 1) and (BarycentricCoordinatesTarget[2] >= 0 and BarycentricCoordinatesTarget[2] <= 1)):
                        PixelLocSrc = np.matmul(
                            AMatrix[triangle_number], BarycentricCoordinatesTarget)
                        PixelLocSrc = np.divide(PixelLocSrc, PixelLocSrc[2])
                        TargetImg1[y][x] = weight*SrcImg[int(
                            PixelLocSrc[1])][int(PixelLocSrc[0])] + (1-weight)*TargetImg1[y][x]
            triangle_number += 1

        # self.HelperFunctions.ShowImage(TargetImg1, 'Target Image')
        FaceSwapOP = self.FaceSwapFunctions.Blending(
            TargetImg1, CloneSrcImg, Mask)
        if debug:
            self.HelperFunctions.ShowImage(TargetImg, 'Target Image')
            self.HelperFunctions.ShowImage(CloneSrcImg, 'Source Image')
            FaceSwapOP = cv2.resize(FaceSwapOP, (720, 720))
            self.HelperFunctions.ShowImage(
                FaceSwapOP, 'FaceSwap using Delaunay Triangulation')
        return FaceSwapOP

    def FaceWarpByTPS(self, frame):
        TargetImg = frame
        SrcImg = frame.copy()
        CloneSrcImg = TargetImg.copy()

        isFace1, _, LandMarkPoints = self.FaceSwapFunctions.DetectFacialLandmarks2(
            TargetImg)
        if(not isFace1 or len(LandMarkPoints) < 2):
            return TargetImg
        LandMarkPointsTarget = LandMarkPoints[0]
        LandMarkPointsSrc = LandMarkPoints[1]
        p = len(LandMarkPointsTarget)
        K = np.zeros([p, p])

        # K matrix
        for idX in range(p):
            for idY in range(p):
                K[idX][idY] = self.HelperFunctions.FuncUofR(
                    np.linalg.norm(np.array(LandMarkPointsTarget[idX]) - np.array(LandMarkPointsTarget[idY]), ord=2))

        # P matrix
        P = np.zeros([p, 3])
        for i, (x, y) in enumerate(LandMarkPointsTarget):
            P[i] = (x, y, 1)

        # lambda I matrix
        I = np.identity(p+3)
        lamb = 1e-4

        # V matrix
        vx = np.zeros([p+3])
        vy = np.zeros([p+3])

        for i, (x, y) in enumerate(LandMarkPointsSrc):
            vx[i] = x
            vy[i] = y
        vx = np.reshape(vx, (p+3, 1))
        vy = np.reshape(vy, (p+3, 1))

        t1 = np.hstack((K, P))
        t2 = np.hstack((np.transpose(P), np.zeros([3, 3])))
        M = np.vstack((t1, t2))

        resultX = np.dot(np.linalg.inv(M + lamb*I), vx)
        resultY = np.dot(np.linalg.inv(M + lamb*I), vy)

        wX = resultX[0:p]
        axX = resultX[p]
        axY = resultX[p+1]
        ax1 = resultX[p+2]

        wY = resultY[0:p]
        ayX = resultY[p]
        ayY = resultY[p+1]
        ay1 = resultY[p+2]

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)
        if debug:
            print("Points inside convex hull: {}".format(hull))
        Mask = np.zeros(
            (TargetImg.shape[0], TargetImg.shape[1], 3), np.uint8)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = Mask[:, :, 1]
        ptsY = np.where(Mask == 255)[0]
        ptsX = np.where(Mask == 255)[1]
        ptsY = np.transpose(ptsY)
        ptsX = np.transpose(ptsX)

        pts = np.vstack((ptsX, ptsY))
        pts = np.transpose(pts)

        if debug:
            print("K: {}".format(K))
            print("P: {}".format(P))
            print("M: {}".format(M))
            print("vx: {}".format(vx))
            print("vy: {}".format(vy))
            print("resultX: {}".format(resultX))
            print("resultY: {}".format(resultY))
            self.HelperFunctions.ShowImage(Mask, 'Convex hull for face')

        for Lpts in pts:
            U1 = (points - Lpts)
            U1 = np.linalg.norm(U1, ord=2, axis=1)
            U1 = self.HelperFunctions.FuncUofR(U1)
            wUX = np.matmul(np.transpose(wX), U1)
            wUY = np.matmul(np.transpose(wY), U1)
            fX = int(ax1 + axX*Lpts[0] + axY*Lpts[1] + wUX)
            fY = int(ay1 + ayX*Lpts[0] + ayY*Lpts[1] + wUY)
            if fX < 0 or fY < 0:
                continue
            TargetImg[Lpts[1]][Lpts[0]] = SrcImg[fY][fX]

        TargetImg1 = self.FaceSwapFunctions.Blending(
            TargetImg, CloneSrcImg, Mask)
        CloneSrcImg = TargetImg1

        ########################
        p = len(LandMarkPointsSrc)
        K = np.zeros([p, p])

        # K matrix
        for idX in range(p):
            for idY in range(p):
                K[idX][idY] = self.HelperFunctions.FuncUofR(
                    np.linalg.norm(np.array(LandMarkPointsSrc[idX]) - np.array(LandMarkPointsSrc[idY]), ord=2))

        # P matrix
        P = np.zeros([p, 3])
        for i, (x, y) in enumerate(LandMarkPointsSrc):
            P[i] = (x, y, 1)

        # lambda I matrix
        I = np.identity(p+3)
        lamb = 1e-4

        # V matrix
        vx = np.zeros([p+3])
        vy = np.zeros([p+3])

        for i, (x, y) in enumerate(LandMarkPointsTarget):
            vx[i] = x
            vy[i] = y
        vx = np.reshape(vx, (p+3, 1))
        vy = np.reshape(vy, (p+3, 1))

        t1 = np.hstack((K, P))
        t2 = np.hstack((np.transpose(P), np.zeros([3, 3])))
        M = np.vstack((t1, t2))

        resultX = np.dot(np.linalg.inv(M + lamb*I), vx)
        resultY = np.dot(np.linalg.inv(M + lamb*I), vy)

        wX = resultX[0:p]
        axX = resultX[p]
        axY = resultX[p+1]
        ax1 = resultX[p+2]

        wY = resultY[0:p]
        ayX = resultY[p]
        ayY = resultY[p+1]
        ay1 = resultY[p+2]

        points = np.array(LandMarkPointsSrc)
        hull = cv2.convexHull(points)
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)
        if debug:
            print("Points inside convex hull: {}".format(hull))
        Mask = np.zeros(
            (TargetImg1.shape[0], TargetImg1.shape[1], 3), np.uint8)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = Mask[:, :, 1]
        ptsY = np.where(Mask == 255)[0]
        ptsX = np.where(Mask == 255)[1]
        ptsY = np.transpose(ptsY)
        ptsX = np.transpose(ptsX)

        pts = np.vstack((ptsX, ptsY))
        pts = np.transpose(pts)

        if debug:
            print("K: {}".format(K))
            print("P: {}".format(P))
            print("M: {}".format(M))
            print("vx: {}".format(vx))
            print("vy: {}".format(vy))
            print("resultX: {}".format(resultX))
            print("resultY: {}".format(resultY))
            self.HelperFunctions.ShowImage(Mask, 'Convex hull for face')

        for Lpts in pts:
            U1 = (points - Lpts)
            U1 = np.linalg.norm(U1, ord=2, axis=1)
            U1 = self.HelperFunctions.FuncUofR(U1)
            wUX = np.matmul(np.transpose(wX), U1)
            wUY = np.matmul(np.transpose(wY), U1)
            fX = int(ax1 + axX*Lpts[0] + axY*Lpts[1] + wUX)
            fY = int(ay1 + ayX*Lpts[0] + ayY*Lpts[1] + wUY)
            if fX < 0 or fY < 0:
                continue
            TargetImg1[Lpts[1]][Lpts[0]] = SrcImg[fY][fX]

        FaceSwapOP = self.FaceSwapFunctions.Blending(
            TargetImg1, CloneSrcImg, Mask)

        # self.HelperFunctions.ShowImage(
        #     FaceSwapOP, 'FaceSwap using Thin Plate Spline')
        return FaceSwapOP
