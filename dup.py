import cv2
#import encoder
import face_recognition
import pickle


model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")         #importing haarcascde model
webcam = cv2.VideoCapture(0)                                                 #for video capturing and store video data



print("Loading Encode File...")
file = open("Encode_File.p",'rb')

encode_id = pickle.load(file)
file.close
encoded_num,studentId = encode_id
#print(studentId)
print("Encoded File Loaded")


while True:
    work,vid = webcam.read()                           #work for True condition and vid for read the information from webcam
    vidS = cv2.resize(vid,(0,0), None, 0.5, 0.5)
    
    vid = cv2.flip(vidS, 1)                             #for mirror video
    bnw = cv2.cvtColor(vid,cv2.COLOR_BGR2RGB)         #changing vid from BGR to GRAY
    
    face = model.detectMultiScale(bnw,scaleFactor = 1.1, minNeighbors=5)    #here the library analyze the grey vid
    face_compare = face_recognition.face_locations(bnw)
    encode_compare = face_recognition.face_encodings(bnw, face_compare)
    
    for encodeFace, faceloc in zip(encode_compare,face_compare):
        matches = face_recognition.compare_faces(encoded_num,encodeFace)
        face_dis = face_recognition.face_distance(encoded_num,encodeFace)
        print("mat:",matches)
        print("faceDis:",face_dis)
    cv2.imshow("Video",vid)     #it displays the video
    #print(face)
    
    key = cv2.waitKey(1)        #to end the program
    
    if key == 32: 
        quit()
    