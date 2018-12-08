import cv2
import os
import numpy as np
from sqlite3 import *
from PIL import Image

class Face:
    def __init__(self):
        self.conn = connect("Database.db")
        self.cursor = self.conn.cursor()
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cam = cv2.VideoCapture(0)
        self.recognizer =  cv2.face.LBPHFaceRecognizer_create()
        try:
            "if previous data is available then load else pass"
            self.recognizer.read('recognizer/trainingdata.yml')
        except:
            pass
        
        try:
            "if table doesn't exist Create the table"
            self.cursor.execute("CREATE TABLE FaceRecognition(id INTEGER PRIMARY KEY, name varchar(50))")
        except:
            pass
        
    def register(self):
        name = input("Enter The Name :")
        self.cursor.execute("insert into FaceRecognition (name) values ('{0}')".format(name))
        Id = self.cursor.execute('select Id from FaceRecognition').fetchall()[-1][0]
        sampleNumber = 0
        print('Please smile')
        while(True):
            img = self.cam.read()[1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sampleNumber += 1
                cv2.imwrite('dataset/User.'+str(Id)+'.'+str(sampleNumber)+'.jpg',gray[y:y+h , x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.waitKey(100)
            cv2.imshow("Face",img)
            if sampleNumber == 20:
                print('Done')
                break
        cv2.destroyAllWindows()
        self.train()

    def train(self):
        path = 'dataset'
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow('training', faceNp)
            cv2.waitKey(10)

        self.recognizer.train(faces,np.array(IDs))
        self.recognizer.save('recognizer/trainingdata.yml')
        cv2.destroyAllWindows()

    

    def detect(self):
        while(True):
            img = self.cam.read()[1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img,(x, y),(x+w, y+h),(0, 0, 255), 2)
                Id, conf=self.recognizer.predict(gray[y:y+h,x:x+w])
                z=self.cursor.execute("select name from FaceRecognition where Id is {0}".format(Id))
                z=z.fetchone()
                if z:
                    Id = str(z[0])
                else:
                    Id = "Unknown"
                cv2.putText(img,str(Id),(x,y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)    

            cv2.imshow("Face",img)
            if cv2.waitKey(1) == 27:
                break

    def __del__(self):
        self.conn.commit()
        self.cursor.close()
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    algo = Face()
    choice = int(input("1 -> Register\n2 -> Detection\n3 -> Exit\n\nChoice...."))
    if choice == 1:
       algo.register()
       algo.detect()
       del(algo)
            
    elif choice == 2:
       algo.detect()
       del(algo)
           
    elif choice == 3:
        del(algo)
            
    else:
       print('Wrong Choice')
