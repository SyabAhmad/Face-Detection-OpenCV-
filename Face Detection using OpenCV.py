import cv2
import matplotlib.pyplot as plt


# You can find this Cascade Classifiers in the README.md file of this repository.
faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
smileCascade = cv2.CascadeClassifier("data/haarcascade_smile.xml")

def detectionOfFace(imgURL):
    img = cv2.imread(imgURL)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for finding faces
    faces = faceCascade.detectMultiScale(gray, 1.2, 2)

    if len(faces) == 0:
        return img
    else:

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # for finding eyes
        eyes = eyeCascade.detectMultiScale(gray, 1.2,2)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img, (ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        # for finding smiles
        smile = smileCascade.detectMultiScale(gray, 1.4,4)

        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(img, (sx,sy),(sx+sw,sy+sh),(255,0,255),1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


imgURL = 'face 1.jpeg'
detectionOfFace(imgURL)
