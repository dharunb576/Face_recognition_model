import cv2
import face_recognition
import pickle
import os

folder = 'img'                         #importing the image folder to variable 'folder'
pathlist = os.listdir(folder)       #to collect all the imges from the img folder which is present inside the "folder"n var
print(pathlist)
imglist = []                                #array variable for storing the images
studentId = []                              #array variable for storing student ids


for path in pathlist:                       #loop for storing the studentIds
    imglist.append(cv2.imread(os.path.join(folder,path)))
    #print(path)
    #print(os.path.splitext(path)[0])
    studentId.append(os.path.splitext(path)[0])
    
#print(len(imglist))
print(studentId)


def findencode(imageslist):                 #to  create a encoded code for all stored images
    encodelist = []
    for img in imageslist:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
        
    return encodelist
print("encoding Started...")
encoded_num = findencode(imglist)           #here encoding the collection of img by using 'def fincencde'
encode_id = [encoded_num,studentId]         #combine face code with the studend_id
print(encoded_num)                          #print the encoded form of img
print("completed!!!")



file = open("Encode_File.p",'wb')           #creating file Named "Encode_file.p" and w-- write mode // b-- binary data storage which mean write in binary code
pickle.dump(encode_id,file)                 
file.close()
print("file Saved")

