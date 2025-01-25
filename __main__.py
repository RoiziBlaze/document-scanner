import cv2
import numpy as np
import easyocr

from fpdf import FPDF

def binar_function(image): # Применение адаптивной бинаризации

    ret3, imgAdaptiveThre = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Применяем пороговую обработку (метод Оцу)

    return imgAdaptiveThre

def biggestContour(contours): # Ищем наибольший контур
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def drawRectangle(img, biggest, thickness): # Рисуем прямоугольник по периметру листа при удачном распознавании углов листа
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img

def reorder(myPoints): # Присвоение индексов точкам (верхний левый угол - один индекс; верхний правый угол - второй индекс и т.д.)
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def savingImg(extention):
    cv2.imwrite("Scanned/myImage" + str(count) + extention, img_result)

    with open("Scanned/imgScannedText" + str(count) + ".txt", "w") as my_file:
        my_file.write(file_text)

    print("Ssved as: 'myImage" + str(count) + extention + "' and " + "imgScannedText" + str(count) + ".txt'")


########################################################################

webCamFeed = True
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 960  # heightImg = 640 (2160)
widthImg = 720  # widthImg = 480 (1620)
reader = easyocr.Reader(['ru'])

is_scanned = False

count = 0

img = np.zeros((heightImg, widthImg, 3), np.uint8)
imgAdaptiveThre = img.copy()

show_text = True
show_threshold = True

while True:
    key = cv2.waitKey(1) & 0xFF

    if webCamFeed:
        success, img = cap.read()
        img = cv2.resize(img, (widthImg, heightImg))  # Изменяем размеры (для удобства вывода окна с итоговым изображением)

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Перевод в градации серого
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)  # Примменяем размытие изображения по Гауссу
        imgThreshold = cv2.Canny(imgBlur, 100, 200)  # Ищем контуры

        kernel = np.ones((3, 3))
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # Расширение линий контура, чтобы исключить их разрывы
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

        # Ищем все контуры на изображении
        imgContours = cv2.cvtColor(imgGray,cv2.COLOR_GRAY2BGR)
        imgBigContour = imgContours.copy()
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Ищем контуры, убирая текст из внутренней части страницы при помощи иерархии
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # Рисуем обнаруженные контуры на 3-м фото

        # Находим самый большой контур
        angles, maxArea = biggestContour(contours)  # Ищем наибольший контур, angles содержит координаты 4 точек - углы листа
        if angles.size != 0:
            angles = reorder(angles)
            cv2.drawContours(imgBigContour, angles, -1, (0, 255, 0), 20)  # Размещаем все найденные точки на картинке
            img_result = drawRectangle(imgBigContour, angles, 2)  # Рисуем прямоугольник по этим координатам
        else:
            img_result = imgBigContour.copy()

    if not webCamFeed:
        if not is_scanned and angles.size != 0:
            pts1 = np.float32(angles)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # Располагаем точки на своих местах
            matrix = cv2.getPerspectiveTransform(pts1, pts2)

            imgWarp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Убираем 20 пикселей с каждой стороны, которые могут содержать лишний фон
            imgWarp = imgWarp[20:imgWarp.shape[0] - 20, 20:imgWarp.shape[1] - 20]
            imgWarp = cv2.resize(imgWarp, (widthImg, heightImg))

            imgWarp = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre = binar_function(imgWarp)
        elif not is_scanned:
            imgWarp = imgGray.copy()
            imgAdaptiveThre = binar_function(imgWarp)

        if not is_scanned:

            text = reader.readtext(imgAdaptiveThre)

            file_text = ""

            for t in text:
                print(t)

                bbox, text_2, score = t
                file_text = file_text + " " + text_2
                bbox = np.array(bbox).astype('int')

            imgAdaptiveThre = cv2.cvtColor(imgAdaptiveThre, cv2.COLOR_BGR2RGB)

            print("Scanning has been finished")
            is_scanned = True

        if not show_threshold:
            img_result = imgWarp.copy()
            file_extension = ".jpg"
        else:
            img_result = imgAdaptiveThre.copy()
            file_extension = ".png"

        if key == ord('s'): # Клавиша "s" - сохранение файлов в .png/.jpg и .txt
            savingImg(file_extension)
            count += 1

        elif key == ord('w'): # Клавиша "w" - сохранение файла в формате .webp и .txt
            savingImg(".webp")
            count += 1

        elif key == ord('p'): # Клавиша "p" - сохранение файла в формате .pdf

            pdf = FPDF('P', 'mm', 'Letter')
            pdf.add_page()

            cv2.imwrite("Temp/temp_image" + file_extension, img_result)
            pdf.image("Temp/temp_image" + file_extension, 2, 0, 210, 297)

            pdf.add_page()

            pdf.add_font('TimesNewRoman', '', 'TimesNewRoman.ttf', True) # В папке с проектом должен находиться файл шрифта TimesNewRoman.ttf
            pdf.set_font('TimesNewRoman', '', 14)
            pdf.cell(40, 10, file_text, ln=True) # Добавляем текст в файл

            pdf.output('Scanned\my_pdf.pdf')
            print("Saved as: 'my_pdf.pdf'")

            count += 1

        if show_text:
            for t in text:
                bbox, text_2, score = t

                bbox = np.array(bbox).astype('int')

                cv2.rectangle(img_result, bbox[0], bbox[2], (0, 255, 0), 1)
                cv2.putText(img_result, text_2, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    cv2.imshow("Result", img_result)

# Управление:

    if key == ord('1'):  # Клавиша "1" - сфотографировать документ и начать его сканирование
        print("Scanning...")
        webCamFeed = False
        is_scanned = False

    elif key == ord('2'):  # Клавиша "2" - сбросить текущий снимок и сделать новое фото
        webCamFeed = True
        is_scanned = True

    elif key == ord('3'): # Клавиша "3" - вкл/выкл показ текста
        show_text = not show_text

    elif key == ord('4'): # Клавиша "4" -  вкл/выкл пороговую обработку
        show_threshold = not show_threshold

    elif key == ord('0'):  # Клавиша "0" - выход из программы
        break