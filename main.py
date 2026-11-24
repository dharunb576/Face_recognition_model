import cv2
#import encoder
import face_recognition
import pickle
import numpy as nmp


#model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")         #importing haarcascde model
webcam = cv2.VideoCapture(0)               #for video capturing and store video data
#webcam = cv2.VideoCapture("https://10.242.70.80:8080/video")

#print("Loading Encode File...")
file = open("Encode_File.p",'rb')

encode_id = pickle.load(file)
file.close
encoded_num,studentId = encode_id
#print(studentId)
#print("Encoded File Loaded")
again = '0'
while True:
    work,vid = webcam.read()                           #work for True condition and vid for read the information from webcam
    vidS = cv2.resize(vid,(640,480), None, 0.25, 0.25)
    
    vid = cv2.flip(vidS, 1)                             #for mirror video
    bnw = cv2.cvtColor(vid,cv2.COLOR_BGR2RGB)         #changing vid from BGR to GRAY
    
   # face = model.detectMultiScale(bnw,scaleFactor = 1.1, minNeighbors=5)    #here the library analyze the grey vid
    face_compare = face_recognition.face_locations(bnw)
    encode_compare = face_recognition.face_encodings(bnw, face_compare)
    
    for encodeFace, faceloc in zip(encode_compare,face_compare):
        matches = face_recognition.compare_faces(encoded_num,encodeFace)
        face_dis = face_recognition.face_distance(encoded_num,encodeFace)
        #print("Match is True or false :",matches)
        #print("faceDis:",face_dis)
        
        match_index = nmp.argmin(face_dis)
        #print("Match Index",matches)
        
        if matches[match_index]:
            
            top, right, bottom, left = faceloc
            cv2.rectangle(vid, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(vid,studentId[match_index], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2, cv2.LINE_AA)
            
        if(again != studentId[match_index]):
            print(studentId[match_index]+" is Present")
            again = studentId[match_index]
        # else:
        #     print("Already Noted")
            
        
    cv2.imshow("Video",vid)     #it displays the video

    
    key = cv2.waitKey(5)        #to end the program
    
    if key == 32: 
        quit()
    