import cv2
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam,hCam=640,480

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList) !=0:
        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img,(x1,y1),15,(200,100,0),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (200, 100, 0), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(0,100,255),3)

        length=math.hypot(x2-x1,y2-y1)
        vol=np.interp(length,[30,200],[minVol,maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        print(int(length),vol)

        cv2.imshow('image',img)
        cv2.waitKey(1)


