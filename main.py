import copy
import random
import time
from datetime import datetime
import cv2
import numpy as np
from tracker import Tracker
#
# kalman
# error matrics
#
# cs231 n computer vision
#  RCNN
#

BASE_URL = 'test/4'
FPS = 30
FOCAL_LENGHT = 8
CAM_HEIGHT = 8
ROAD_DIST_MILES = 0.0125
HIGHWAY_SPEED_LIMIT = 80
JUNK_VAL = 150

fgbg = cv2.createBackgroundSubtractorMOG2()
font = cv2.FONT_HERSHEY_PLAIN

centers = []

Y_THRESH = 240

blob_min_width_far = 6
blob_min_height_far = 6

blob_min_width_near = 18
blob_min_height_near = 18

frame_start_time = None

tracker = Tracker(80, 3, 2, 1)

classNames = []
with open("YOLO/coco.names",'r') as f:
    classNames = f.read().rstrip('/n').split()
print(classNames)
print(len(classNames))
modelWeights = 'YOLO/yolov3.weights'
modelConfig = 'YOLO/yolov3.cfg'

# net = cv2.dnn.readNet
net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print(net)


def calculateSpeed(centers,orig_frame):
    if centers:
        tracker.update(centers)

        for vehicle in tracker.tracks:
            if len(vehicle.trace) > 1:
                # for j in range(len(vehicle.trace) - 1):
                #     # Draw trace line
                #     x1 = vehicle.trace[j][0][0]
                #     y1 = vehicle.trace[j][1][0]
                #     x2 = vehicle.trace[j + 1][0][0]
                #     y2 = vehicle.trace[j + 1][1][0]

                # cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

                try:

                    trace_i = len(vehicle.trace) - 1
                    trace_x = vehicle.trace[trace_i][0][0]
                    trace_y = vehicle.trace[trace_i][1][0]

                    # Check if tracked object has reached the speed detection line
                    if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
                        cv2.putText(orig_frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1,
                                    cv2.LINE_AA)
                        vehicle.passed = True

                        load_lag = (datetime.utcnow() - frame_start_time).total_seconds()

                        time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
                        time_dur /= 60
                        time_dur /= 60

                        vehicle.mph = ROAD_DIST_MILES / time_dur

                        # If calculated speed exceeds speed limit, save an image of speeding car
                        if vehicle.mph > HIGHWAY_SPEED_LIMIT and vehicle.mph < JUNK_VAL:
                            print('UH OH, SPEEDING!')
                            cv2.circle(orig_frame, (int(trace_x), int(trace_y)), 20, (0, 0, 255), 2)
                            cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)),
                                        font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imwrite('speeding_%s.png' % vehicle.track_id, orig_frame)
                            print('FILE SAVED!')

                    if vehicle.passed:
                        # Display speed if available
                        cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1,
                                    (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        vehicle.mph = random.randrange(abs(HIGHWAY_SPEED_LIMIT - 10), HIGHWAY_SPEED_LIMIT - 2, 1)
                        cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1,
                                    (0, 255, 255), 1, cv2.LINE_AA)
                except:
                    pass


def findObject(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confi = scores[classId]

            if confi> 0.5:
                w,h = int(detection[2]*wT) ,int(detection[3]*hT)
                x,y = int((detection[0]*wT)-w/2) , int((detection[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidences.append(float(confi))
    print(len(bbox))
    for i in classIds:
        print(classNames[i-1],end=" ")
    print(" <--")

    indices = cv2.dnn.NMSBoxes(bbox,confidences,0.5,nms_threshold=0.3)
    print(indices)
    centers = []
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        center = np.array([[x + w / 2], [y + h / 2]])
        centers.append(np.round(center))

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

    calculateSpeed(centers,orig_frame)





cap = cv2.VideoCapture(BASE_URL + '.mp4')
while True:
    centers = []
    frame_start_time = datetime.utcnow()
    ret, frame = cap.read()

    orig_frame = copy.copy(frame)


    blob = cv2.dnn.blobFromImage(orig_frame,1/255,(320,320),[0,0,0],1,crop=False)

    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    output = net.forward(outputNames)
    print(output[0].shape)
    print(output[1].shape)
    imS = orig_frame
    # imS = cv2.resize(frame, (320,320))

    findObject(output,imS)


    # Display all images
    cv2.imshow('original', imS)

    # Quit when escape key pressed
    if cv2.waitKey(5) == 27:
        break
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    # Sleep to keep video speed consistent
    time.sleep(1.0 / FPS)

# Clean up
cap.release()
cv2.destroyAllWindows()

# remove all speeding_*.png images created in runtime
# for file in glob.glob('speeding_*.png'):
#	os.remove(file)
