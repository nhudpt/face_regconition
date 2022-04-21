import cv2
import os
from facenet.face_contrib import *

def start_capture(name, ID):
        path = "./data/" + name + '_' +  ID
        num_of_images = 0
        try:
            os.makedirs(path)
        except:
            print('Directory Already Created')
        vid = cv2.VideoCapture(0)
        face_recognition = Recognition('models', 'models/your_model.pkl')
        while True:
            ret, img = vid.read()
            new_img = None
            faces = face_recognition.identify(img)
            colors = np.random.uniform(0, 255, size=(1, 3))
            if faces is not None:
                for idx, face in enumerate(faces):
                    face_bb = face.bounding_box.astype(int)
                    cv2.rectangle(img, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
                    cv2.putText(img, "Face Detected", (face_bb[0], face_bb[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                    cv2.putText(img, str(str(num_of_images) + " images captured"), (face_bb[0], face_bb[1] + face_bb[3] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                    new_img = img[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2]]
            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF
            try :
                cv2.imwrite(str(path+"/"+name+"_"+str(num_of_images)+".jpg"), new_img)
                num_of_images += 1
            except :
                pass
            if key == ord("q") or key == 27 or num_of_images > 30:
                break
        cv2.destroyAllWindows()
        return num_of_images

if __name__ == '__main__':
    name = input ('Ho va ten: ')
    ID = input ('ID: ')
    start_capture (name, ID)