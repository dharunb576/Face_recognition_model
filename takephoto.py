# import cv2
# import os

# model_ph = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# def face_cropped(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = model_ph.detectMultiScale(gray, 1.3, 5)
    
#     face_cropped = img[y:y + h, x:x + w]
#     return face_cropped

# try:
#     cap = cv2.VideoCapture (0)
#     img_id = 0
#     id = 1
#     while True:
#         ret, my_frame = cap.read()
#         if face_cropped (my_frame) is not None:
#             img_id += 1
#             face = cv2.resize(face_cropped(my_frame), (450, 450))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             file_name_path = "data/user." + str(id) + "." + str(img_id) + ".jpg"
#             cv2.imwrite(file_name_path, face)
#             cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX)
#             cv2.imshow("Cropped Face", face)
            
#         if cv2.waitKey(1) == 13 or int(img_id) == 100:
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Completed")
# except Exception as es:
#     print(str(es))


import cv2
import os

# Load Haar Cascade
model_ph = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Folder to save photos
save_path = r"F:\python Project\Facial Recognization\photo"
os.makedirs(save_path, exist_ok=True)  # Create folder if it doesn't exist

# Function to crop face
def face_cropped(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model_ph.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Take the first face detected
    x, y, w, h = faces[0]
    return img[y:y + h, x:x + w]

try:
    cap = cv2.VideoCapture(0)
    img_id = 0
    user_id = 1

    while True:
        ret, my_frame = cap.read()
        if not ret:
            break

        face = face_cropped(my_frame)
        if face is not None:
            img_id += 1
            face = cv2.resize(face, (450, 450))
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save the image
            file_name_path = os.path.join(save_path, f"user.{user_id}.{img_id}.jpg")
            cv2.imwrite(file_name_path, face_gray)

            # Display the face with img_id
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face", face)

        # Exit on Enter key or after 100 images
        if cv2.waitKey(1) == 13 or img_id == 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Completed")

except Exception as es:
    print(str(es))
