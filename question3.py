#Programa deve somente utilizar a detecção dos sorriso, em que deve desenhar um retângulo no sorriso encontrado, e indicar quantos sorrisos foram detectados escrevendo "N Sorrisos detectados" com o método cv2.putText().

import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

captura = cv2.VideoCapture(0)

while(1):
    if captura.isOpened():
        ret, frame = captura.read()
    else:
        frame = cv2.imread('asd.jpg')

    faces = face_classifier.detectMultiScale(frame, 1.0485258, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)    

        rosto = frame[y:y + h, x:x + w]
        
        smiles = smile_classifier.detectMultiScale(rosto, 1.06, 100)

        frame = cv2.putText(frame, str(len(smiles)) + ' sorrisos detectados', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for (ex, ey, ew, eh) in smiles:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('faces', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
