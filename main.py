import cv2 as cv
import numpy as np
import time
from dataclasses import dataclass
from pynput.keyboard import Controller
from HandTrackingModule import handDetector
# from cvzone import cornerRect


def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv.line(img, (x, y), (x + l, y), colorC, t)
    cv.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv.line(img, (x, y1), (x + l, y1), colorC, t)
    cv.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv.line(img, (x1, y1), (x1, y1 - l), colorC, t)
    return img


cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == 640
assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == 480

detector = handDetector(detectionCon=0.8)
keys = (("Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"),
        ("A", "S", "D", "F", "G", "H", "J", "K", "L", ";"),
        ("Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"),
        ("<", " "))
finalText = ""
keyboard = Controller()
DX, DY = 0, 40


def drawALL(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cornerRect(imgNew, (x, y, w, h), 20, rt=0)
        cv.rectangle(imgNew, button.pos, (x + w, y + h), (255, 0, 255), cv.FILLED)
        cv.putText(imgNew, button.text, (x + DX, y + DY), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out


@dataclass
class Button:
    pos: list
    text: str
    size: list


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100//2 * j + 50, 100//2 * i + 50], key, [85//2, 85//2]))

while cap.isOpened():
    _, img = cap.read()
    img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)

    img = drawALL(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                cv.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv.FILLED)
                cv.putText(img, button.text, (x + DX, y + DY), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                l, _, _ =  detector.findDistance(8, 12, img, draw=False)

                ## when clicked
                if l < 40:
                    cv.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv.FILLED)
                    cv.putText(img, button.text, (x + DX, y + DY), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if button.text == "<":
                        finalText = finalText[:-1]
                        keyboard.press('\010')
                    else:
                        finalText += button.text
                        keyboard.press(button.text)
                    time.sleep(0.15)

    cv.rectangle(img, (50, 710//2), (700//3*2, 610//2), (175, 0, 175), cv.FILLED)
    cv.putText(img, finalText, (60, 690//2), cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
                          
    cv.imshow("Keyboard", img)
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
