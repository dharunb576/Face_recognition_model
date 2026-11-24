# import cv2

# img = cv2.imread('img.jpg')
# grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Bala',grey)
# key = cv2.waitKey()
# if key == 32:
#     quit()



import cv2
print("OpenCV version:", cv2.__version__)

import face_recognition
import dlib
print("face_recognition and dlib installed successfully!")