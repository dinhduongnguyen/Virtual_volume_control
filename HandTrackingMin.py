import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

while True:
    success,img=cap.read()
 #   cv2.rectangle(img, (100, 100), (200, 200), [255, 0, 0], 2)
    img=cv2.flip(img,1)
    imRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)



    cv2.imshow('image', img)
    cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
