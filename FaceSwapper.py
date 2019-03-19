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
from Misc.FaceSwapUtils import FaceSwapUtils
from Misc.FaceSwapMethods import FaceSwapImages, FaceSwapVideoWithImage, FaceSwapVideoFaces


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Mode', default='1',
                        help='FaceSwap Mode. 0: Swap faces in two images. 1: Swap largest face in video with user defined image. 2: Swap two faces in single video, Default:0')
    Parser.add_argument('--FaceSwapMethod', default='1',
                        help='FaceSwap FaceSwapMethod. 0: Triangulation. 1: Thin Plate Spline, Default:0')
    Parser.add_argument('--Path1', default='/home/rohith/CMSC733/git/FaceSwap/Data/rohitk.webm',
                        help='Video path for Mode 1 and 2. Target image for Mode 0, Default:/home/rohith/CMSC733/git/FaceSwap/Data/elon.mp4')
    Parser.add_argument('--Path2', default='/home/rohith/CMSC733/git/FaceSwap/Data/rohith.jpeg',
                        help='Image path to be swapped for Mode 0 and 1. Dont care for Mode2, Default:/home/rohith/CMSC733/git/FaceSwap/Data/padme2.jpg')
    Parser.add_argument('--ModelPath', default='/home/rohith/CMSC733/git/FaceSwap/Models/shape_predictor_68_face_landmarks.dat',
                        help='Model path of dlib predictor, Default:/home/rohith/CMSC733/git/FaceSwap/Models/shape_predictor_68_face_landmarks.dat')

    Args = Parser.parse_args()
    Mode = Args.Mode
    FaceSwapMethod = Args.FaceSwapMethod
    Path1 = Args.Path1
    Path2 = Args.Path2
    ModelPath = Args.ModelPath
    HelperFunctions = ImageUtils()

    if Mode == '0':
        if FaceSwapMethod == '0':
            print(
                "**********Triangulation method started for FaceSwap in two images**********")
            swapFacesTri = FaceSwapImages(
                Path1, Path2, ModelPath)
            faceSwapOP = swapFacesTri.FaceWarpByTriangulation(1)
            HelperFunctions.ShowImage(
                faceSwapOP, 'Triangulation method FaceSwap')
            print(
                "**********Triangulation method ended for FaceSwap in two images**********")
        elif FaceSwapMethod == '1':
            print("**********TPS method started for FaceSwap in two images**********")
            swapFacesTPS = FaceSwapImages(
                Path1, Path2, ModelPath)
            faceSwapOP = swapFacesTPS.FaceWarpByTPS()
            HelperFunctions.ShowImage(faceSwapOP, 'TPS method FaceSwap')
            print("**********TPS method ended for FaceSwap in two images**********")

    elif Mode == '1':
        cap = cv2.VideoCapture(Path1)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w = int(cap.get(3))
        h = int(cap.get(4))

        if FaceSwapMethod == '0':
            print(
                "**********Triangulation method started for swapping face in video with image**********")
            swapFaces1 = FaceSwapVideoWithImage(Path2, ModelPath)
            out1 = cv2.VideoWriter('DataOutputTri.avi', fourcc, 20.0, (w, h))
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
            cap.release()
            out1.release()
            cv2.destroyAllWindows()
            print(
                "**********Triangulation method ended for swapping face in video with image**********")
        elif FaceSwapMethod == '1':
            print(
                "**********TPS method started for swapping face in video with image**********")
            # i = 0
            swapFaces2 = FaceSwapVideoWithImage(Path2, ModelPath)
            out2 = cv2.VideoWriter('DataOutputTPS.avi', fourcc, 20.0, (w, h))
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    frame = swapFaces2.FaceWarpByTPS(frame)
                    out2.write(frame)
                    # print(i)
                    # i += 1
                    # cv2.imshow('frame', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    break
            cap.release()
            out2.release()
            cv2.destroyAllWindows()
            print(
                "**********TPS method ended for swapping face in video with image**********")

    elif Mode == '2':
        cap = cv2.VideoCapture(Path1)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        w = int(cap.get(3))
        h = int(cap.get(4))
        if FaceSwapMethod == '0':
            print(
                "**********Triangulation method started for swapping face in video**********")
            swapFaces1 = FaceSwapVideoFaces(ModelPath)
            out1 = cv2.VideoWriter('DataOutputTri.avi', fourcc, 20.0, (w, h))
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
            cap.release()
            out1.release()
            cv2.destroyAllWindows()
            print(
                "**********Triangulation method ended for swapping face in video**********")
        elif FaceSwapMethod == '1':
            print(
                "**********TPS method started for swapping face in video**********")
            i = 0
            swapFaces2 = FaceSwapVideoFaces(ModelPath)
            out2 = cv2.VideoWriter('DataOutputTPS.avi', fourcc, 20.0, (w, h))
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    frame = swapFaces2.FaceWarpByTPS(frame)
                    out2.write(frame)
                    # print(i)
                    # i += 1
                    # cv2.imshow('frame', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else:
                    break
            cap.release()
            out2.release()
            cv2.destroyAllWindows()
            print(
                "**********TPS method ended for swapping face in video**********")


if __name__ == '__main__':
    main()
