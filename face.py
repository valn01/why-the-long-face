import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

oclinho = cv2.imread('oculos.png')

captura = cv2.VideoCapture(0)

#hey copilot, help me configure my git user from the terminal, i forgot how to do it
#the command is git config --global user.email "email" and git config --global user.name "name"

while(1):
    ret, frame = captura.read()

    faces = face_classifier.detectMultiScale(frame, 1.0485258, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
        rosto = frame[y:y + h, x:x + w]
        olhos = eye_classifier.detectMultiScale(rosto, 1.0485258, 6)

        olho_esquerdo = {'x':999999,'y':0}
        olho_direito = {'x':0,'y':0}

        for (ex, ey, ew, eh) in olhos:
            olho_x = x + ex
            olho_y = y + ey

            if olho_x < olho_esquerdo['x']:
                olho_esquerdo['x'] = olho_x
                olho_esquerdo['y'] = olho_y
            if (olho_x + ew) > olho_direito['x']:
                olho_direito['x'] = olho_x + ew
                olho_direito['y'] = olho_y + eh

        if olho_esquerdo['x'] < olho_direito['x']:
            largura = olho_direito['x'] - olho_esquerdo['x']
            altura = olho_direito['y'] - olho_esquerdo['y']
            oclinho = cv2.resize(oclinho, (largura, altura))
            frame = cv2.bitwise_and(frame[olho_esquerdo["x"]:largura,olho_esquerdo["y"]:altura],oclinho)#não consegui fazer a adição do png, apenas um retângulo preto(commente a linha atual e descomente a próxima )
#             cv2.rectangle(frame, (olho_esquerdo['x'], olho_esquerdo['y']), (olho_direito['x'], olho_direito['y']), (0, 0, 0), -1)

    cv2.imshow('oclinho daora', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

captura.release()
cv2.destroyAllWindows()
