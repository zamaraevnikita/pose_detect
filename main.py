import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
#from classifypose import classifypose
#from draw import draw_connections,draw_keypoints
from functions import classifypose,draw
interpreter = tf.lite.Interpreter(model_path="./models/lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}
########################################### ДЛЯ КАМЕРЫ

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     re, frame = cap.read()
    
#     image = frame.copy()
#     image = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192, 192)
#     input_image = tf.cast(image, dtype=tf.float32)
    
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
#     interpreter.invoke()
#     keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
#     try:
#         classifypose(frame)
#     except:
#         pass
#     # Rendering 
#     draw.draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
#     draw.draw_keypoints(frame, keypoints_with_scores, 0.3)
#     cv2.imshow('MoveNet Lightning', frame)
    
#     if cv2.waitKey(10) & 0xFF==ord('q'):
#         break
        
# cap.release()
# cv2.destroyAllWindows()
################################################################ ДЛЯ ФОТО
# # Загрузка фотографии
# image = cv2.imread('new.jpg')

# # Преобразование изображения в формат, поддерживаемый моделью
# image = cv2.resize(image, (192, 192))  # Меняем размер изображения на 192x192 (аналогично изменению размера видеокадра)
# input_image = tf.cast(image, dtype=tf.float32)
# input_image = np.expand_dims(input_image, axis=0)  # Добавляем размерность пакета (batch dimension)

# # Установка входных данных для модели
# input_details = interpreter.get_input_details()
# interpreter.set_tensor(input_details[0]['index'], input_image)

# # Выполнение вывода модели
# interpreter.invoke()

# # Получение ключевых точек с оценками уверенности
# output_details = interpreter.get_output_details()
# keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

# # Попытка классифицировать позу (не реализована в предоставленном коде)
# try:
#     classifypose(image)
# except:
#     pass

# # Отрисовка позы на изображении
# draw.draw_connections(image, keypoints_with_scores, EDGES, 0.3)
# draw.draw_keypoints(image, keypoints_with_scores, 0.3)

# # Сохранение результата
# cv2.imwrite('output_image.jpg', image)


############################################################ ДЛЯ ВИДЕО
# Открытие видеофайла
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Преобразование кадра в формат, поддерживаемый моделью
    frame = cv2.resize(frame, (192, 192))
    input_image = tf.cast(frame, dtype=tf.float32)
    input_image = np.expand_dims(input_image, axis=0)
    
    # Установка входных данных для модели
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    # Выполнение вывода модели
    interpreter.invoke()
    
    # Получение ключевых точек с оценками уверенности
    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Попытка классифицировать позу
    try:
        classifypose(frame)
    except:
        pass
    
    # Отрисовка позы на кадре
    draw.draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
    draw.draw_keypoints(frame, keypoints_with_scores, 0.3)
    
    # Показ кадра
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()