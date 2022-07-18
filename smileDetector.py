# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:38:12 2021

@author: jordi
"""

import cv2

##Data de entrenamiento
faceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smileData = cv2.CascadeClassifier("haarcascade_smile.xml")

##Capturar video
webcam = cv2.VideoCapture(0)

##Loop para checar los frames
while True:
    ###Tmamos el framde de la webcam
    frameRead, frame = webcam.read()
    
    ###Checamos que si pudo tomar el frame
    if frameRead:
        gris_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
    ###Detectar caras
    coordinadorCaras = faceData.detectMultiScale(
        gris_img,
        1.04,
        6,
        cv2.CASCADE_SCALE_IMAGE,
        (30,30),
        (380,380),
        )
    """
    coordinadorSonrisa = smileData.detectMultiScale(
        gris_img,
        1.7,
        20,
        cv2.CASCADE_SCALE_IMAGE,
        (30,30),
        (180,180),
        )
    """
    frame2 = frame
    ###Poner los rectangulos en el frame
    for (x,y,w,h) in coordinadorCaras:
        # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
        
        ###Reducimos a solo la imagen de la cara y volvemos gris 
        ###(con numpy N-dimensional array slicing)
        the_face = frame[y:y+h,x:x+w]
        face_gray = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)
        
        coordinadorSonrisa = smileData.detectMultiScale(
        face_gray,
        1.3,
        20,
        cv2.CASCADE_SCALE_IMAGE,
        (10,10),
        (500,500),
        )
        
        if len(coordinadorSonrisa) > 0:
            cv2.putText(frame2,"SMILE", (x,y+h+40), fontScale=3,
            fontFace = cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 255), 3)

        
    ###Mostrar la cámara
    cv2.imshow("Webcam (press Q to exit)",frame2)

    ###Poner sonrisas solo en las caras
    for (x1,y1,w1,h1) in coordinadorSonrisa:
        cv2.rectangle(the_face, (x1,y1), (x1+w1,y1+h1), (0,255,0), 3)

    # for (x,y,w,h) in coordinadorSonrisa:
    #     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    ###Esperar y obtener tecla
    key = cv2.waitKey(1)
    
    ###Salir al presionar q
    if key==81 or key==113:
        break
        

##Release the webcam
webcam.release()
cv2.destroyAllWindows()


print("Code complete")