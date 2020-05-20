import numpy as np
import cv2
import os



class Detectron:
    def __init__(self):
        self.palms_cascade = cv2.CascadeClassifier(os.path.dirname(__file__) + '/palm.xml')
        self.fists_cascade = cv2.CascadeClassifier(os.path.dirname(__file__) + '/fist.xml')
    
        
    def detect(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # possible frame pre-processing:
        # gray_frame = cv2.equalizeHist(gray_frame)
        gray_frame = cv2.medianBlur(gray_frame, 5)

        scaleFactor = 1.5 # range is from 1 to ..
        minNeighbors = 5 # range is from 0 to ..
        flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
        minSize = (50,50) # range is from (0,0) to ..
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        fists = self.fists_cascade.detectMultiScale(
            gray_frame,
            scaleFactor,
            1,
            flag,
            minSize)
        palms = self.palms_cascade.detectMultiScale(
            gray_frame,
            scaleFactor,
            minNeighbors,
            flag,
            minSize)
        
        if len(fists) + len(palms) > 1:
            return []
        
        for f in fists:
            cv2.rectangle(frame, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]), (0, 255, 255), 2)
            return f
            
        for p in palms:
            cv2.rectangle(frame, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), (0, 255, 255), 2)
            return p
        
        return []
'''
if __name__ == "__main__":
    # load pretrained cascades
    detectron = Detectron()
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        b = detectron.detect(frame)
        if len(b) == 4:
            cv2.rectangle(frame, (b[0], b[1]), (b[0]+b[2],b[1]+b[3]), (0, 255, 255), 2)
        cv2.imshow(windowName, cv2.flip(frame, 1))
        if cv2.waitKey(30) >= 0:
            showlive = False

    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()
'''
