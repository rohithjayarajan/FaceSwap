#!/usr/bin/evn python

"""
@file    FaceSwapVideo.py
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


class FaceSwap:
    def __init__(self, Image_Path, ModelPath):
        self.ModelPath = ModelPath
        # ImageTemp = cv2.resize(ImageTemp, (1000, 800))
        self.Image = cv2.imread(Image_Path)
        self.FaceDetector = dlib.get_frontal_face_detector()
        self.ShapePredictor = dlib.shape_predictor(self.ModelPath)
        self.HelperFunctions = ImageUtils()

    def DetectFacialLandmarks(self, InputImage):
        GrayImage = cv2.cvtColor(InputImage, cv2.COLOR_BGR2GRAY)
        dets = self.FaceDetector(GrayImage, 1)
        for (_, det) in enumerate(dets):
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
            return DelaunayBBox, LandMarkPoints

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

    def FaceWarpByTriangulation(self, frame, weight):
        TargetImg = frame
        SrcImg = self.Image
        CloneSrcImg = TargetImg.copy()
        Mask = np.zeros(TargetImg.shape, TargetImg.dtype)

        DelaunayBBoxTarget, LandMarkPointsTarget = self.DetectFacialLandmarks(
            TargetImg)
        DelaunayTrianglesTarget = self.GetDelaunayTriangulation(
            DelaunayBBoxTarget, LandMarkPointsTarget, TargetImg)

        _, LandMarkPointsSrc = self.DetectFacialLandmarks(
            SrcImg)
        DelaunayTrianglesSrc = self.getTriangulationCorrespondence(
            LandMarkPointsTarget, DelaunayTrianglesTarget, LandMarkPointsSrc)

        points = np.array(LandMarkPointsTarget)
        hull = cv2.convexHull(points)
        color = (255, 255, 255)
        cv2.fillPoly(Mask, [hull], color)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)

        _, BInvMatrix = self.getMatrixBary(DelaunayTrianglesTarget)
        AMatrix, _ = self.getMatrixBary(DelaunayTrianglesSrc)

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
        FaceSwapOP = self.Blending(TargetImg, CloneSrcImg, Mask)
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

        _, LandMarkPointsTarget = self.DetectFacialLandmarks(
            TargetImg)
        _, LandMarkPointsSrc = self.DetectFacialLandmarks(
            SrcImg)
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
        lamb = 1e-16

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

        FaceSwapOP = self.Blending(TargetImg, CloneSrcImg, Mask)
        # self.HelperFunctions.ShowImage(
        #     FaceSwapOP, 'FaceSwap using Thin Plate Spline')
        return FaceSwapOP


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Video_Path', default='/home/rohith/CMSC733/git/FaceSwap/Data/elon.mp4',
                        help='Video path, Default:/home/rohith/CMSC733/git/FaceSwap/Data/elon.mp4')
    Parser.add_argument('--Image_Path', default='/home/rohith/CMSC733/git/FaceSwap/Data/padme2.jpeg',
                        help='Image path, Default:/home/rohith/CMSC733/git/FaceSwap/Data/padme2.jpg')
    Parser.add_argument('--ModelPath', default='/home/rohith/CMSC733/git/FaceSwap/Models/shape_predictor_68_face_landmarks.dat',
                        help='Model path of dlib predictor, Default:/home/rohith/CMSC733/git/FaceSwap/Models/shape_predictor_68_face_landmarks.dat')

    Args = Parser.parse_args()
    Video_Path = Args.Video_Path
    Image_Path = Args.Image_Path
    ModelPath = Args.ModelPath
    cap = cv2.VideoCapture(Video_Path)
    w = int(cap.get(3))
    h = int(cap.get(4))

    print("**********Triangulation method started**********")
    swapFaces1 = FaceSwap(Image_Path, ModelPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter('Data3OutputTri.avi', fourcc, 20.0, (w, h))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = swapFaces1.FaceWarpByTriangulation(frame, 1)
            out1.write(frame)

            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out1.release()
    cv2.destroyAllWindows()
    print("**********Triangulation method ended**********")
    i = 0
    print("**********TPS method started**********")
    swapFaces2 = FaceSwap(Image_Path, ModelPath)
    cap = cv2.VideoCapture(Video_Path)
    out2 = cv2.VideoWriter('Data2OutputTPS.avi', fourcc, 20.0, (w, h))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = swapFaces2.FaceWarpByTPS(frame)
            out2.write(frame)
            print(i)
            i += 1
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out2.release()
    cv2.destroyAllWindows()
    print("**********TPS method ended**********")


if __name__ == '__main__':
    main()
