# start of something

import cv2

thres = 0.5  # Threshold to detect object

# img = cv2.imread('test_images/teddy_bear.jpeg')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # ID 3
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # ID 4
cap.set(cv2.CAP_PROP_BRIGHTNESS, 70) # ID 10
# cap.set(cv2.CAP_PROP_CONTRAST,100)
# cap.set(27, 100)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = thres)
    print(classIds, confs, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds, confs, bbox):
                text = '%s: %s' % (classNames[classId-1].upper(), confidence)
                print(text)
                cv2.rectangle(img, box, color = (0, 21, 255) , thickness=2)
                cv2.putText(img, classNames[classId-1].upper() + ' : ' + str(int(confidence*100)) + ' %', (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 21, 255),2)

    cv2.imshow('Object Detection with cv2', img)
    cv2.waitKey(1)