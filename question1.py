#Programa deve somente utilizar a detecção da face, em que deve desenhar um retângulo na face encontrada, e indicar quantas faces foram detectadas escrevendo "N faces detectadas" com o método cv2.putText().

import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

captura = cv2.VideoCapture(0)

while(1):
    if captura.isOpened():
        ret, frame = captura.read()
    else:
        frame = cv2.imread('asd.jpg')

    faces = face_classifier.detectMultiScale(frame, 1.0485258, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)        

        cv2.putText(frame, str(len(faces)) + ' faces detectadas', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        cv2.imshow('faces', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break