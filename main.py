import cv2
import numpy as np
import random
# import inspect

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.legacy.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.legacy.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.legacy.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.legacy.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.legacy.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.legacy.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.legacy.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.legacy.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
 
  return tracker

# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w,h))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()


threshold = 0.45 # threshold to detect object
nms_threshold=0.5

# img = cv2.imread('lena.png')
videoPath = 'istockphoto-497015728-640_adpp_is.mp4' #'less blurry video.mp4'
cap = cv2.VideoCapture(videoPath)
# cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(10, 150)


# tracker = cv2.legacy.TrackerMOSSE_create()
# tracker = cv2.legacy.TrackerCSRT_create()


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

success, frame = cap.read()
# frame = cv2.rotate(frame, cv2.ROTATE_180)
cv2.imshow("Tracking", frame)
# print(inspect.getfullargspec(net.detect))
classIds, confs, bboxes = net.detect(frame, threshold, nms_threshold)

indices = cv2.dnn.NMSBoxes(bboxes, confs, threshold, nms_threshold)
indices = [i for i in indices if classIds[i]==1]
# confs, bboxes = zip(*[(confs[i],bboxes[i]) for i in indices]) if indices else [], []
confs = [confs[i] for i in indices]
bboxes = [list(bboxes[i]) for i in indices]

# bbox = cv2.selectROI("Tracking",frame,False)
# tracker.init(frame, bbox)


# Specify the tracker type
trackerType = "KCF"
 
# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()
# print(multiTracker.__dir__())
# Initialize MultiTracker

frameCount = 0
peopleSeen = 0
pv = 0
trackers = []
def add_tracker(bbox):
  global multiTracker, frameCount, trackers, peopleSeen, frame, trackerType
  trackers.append([peopleSeen])
  peopleSeen += 1
  color=tuple(random.randint(0,255) for _ in range(3))
  trackers[-1].append(color)
  trackers[-1].append(frameCount)
  trackers[-1].append(bbox)
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

def reinit_tracker(bbox):
  global multiTracker, frame, trackerType
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)


for bbox in bboxes:
  add_tracker(bbox)

def drawBox(img, box, label="Tracking", color=(0,255,0)):
  x,y,w,h = map(int,box)
  cv2.rectangle(img, (x,y), (x+w, h+y), color, 2, 1)
  cv2.putText(img, label, (x+25,y+25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
def overlap(box0, box1):
  x0,y0,w0,h0 = box0
  x1,y1,w1,h1 = box1
  xl = max(x0,x1)
  xr = min(x0+w0,x1+w1)
  yl = max(y0,y1)
  yr = min(y0+h0,y1+h1)
  if xl >= xr or yl >= yr: return 0.0
  else:
    hm = 1/(1/w0/h0 + 1/w1/h1)
    return (xr-xl)*(yl-yr) / hm

overlap_min = 0.5

hold_length = 5

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter('output.mp4',fourcc,10.0,(640,360))


while True:
  pv = peopleSeen
  frameCount += 1
  # timer = cv2.getTickCount()
  success, frame = cap.read()
  if not success: break
  # frame = cv2.rotate(frame, cv2.ROTATE_180)
  success, bbox_t = multiTracker.update(frame)
  # print(bbox_t)


  # fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
  # cv2.putText(frame, str(int(fps)), (75,50),
  #     cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)


  classIds, confs, bbox_n = net.detect(frame, confThreshold=threshold)
  
  indices = cv2.dnn.NMSBoxes(bbox_n, confs, threshold, nms_threshold=nms_threshold)
  
  indices = [i for i in indices if classIds[i]==1]
  # confs, bboxes = zip(*[(confs[i],bboxes[i]) for i in indices]) if indices else [], []
  confs = [confs[i] for i in indices]
  bbox_n = [list(bbox_n[i]) for i in indices]
  
  for i,box in enumerate(bbox_t):
    if not bbox_n: break
    if sum(box):
      mo = max(overlap(box,boxn) for boxn in bbox_n)
      if mo >= overlap_min:
        for j,boxn in enumerate(bbox_n):
          if overlap(box,boxn) == mo:
            # trackers[i][-1] = boxn
            break
        bbox_n.pop(j)
        trackers[i][2] = frameCount
        trackers[i][3] = box
        drawBox(frame, box, str(trackers[i][0]), trackers[i][1])
        continue
    mo = max(overlap(trackers[i][3],boxn) for boxn in bbox_n)
    if mo >= overlap_min:
      for j,boxn in enumerate(bbox_n):
        if overlap(trackers[i][3],boxn) == mo:
          trackers[i][2] -= hold_length
          trackers.append([trackers[i][0], trackers[i][1],frameCount, boxn])
          break
      bbox_n.pop(j)
    
  if bbox_n:
    for box in bbox_n:
      add_tracker(box)
    trackers.append([frameCount-hold_length]*4)
  
  if trackers and min([tracker[2] for tracker in trackers]) <= frameCount-hold_length:
    toKeep = [tracker for tracker in trackers if tracker[2] > frameCount-hold_length]
    multiTracker = cv2.legacy.MultiTracker_create()
    trackers = toKeep
    for tracker in trackers:
      reinit_tracker(tracker[-1])


  for tracker in trackers:
    if tracker[2]==frameCount:
      drawBox(frame, tracker[-1], str(tracker[0]), tracker[1])
  

  cv2.putText(frame, "People in frame:" + str(peopleSeen - pv), (15,330),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3,1)

  # for conf, box in zip(confs, bbox_n):
  #   # box = bbox_n[i]
  #   drawBox(frame, box, "", (255,0,0))
  #   # cv2.putText(frame, classNames[classIds[i]-1].upper(), (box[0]+10, box[1]+30),
  #   #       cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
  #   cv2.putText(frame, str(round(conf*100,2)), (box[0]+200, box[1]+30),
  #         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)


  # if len(classIds) != 0:
  #   for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
  #     cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
  #     cv2.putText(frame, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
  #         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
  #     cv2.putText(frame, str(round(confidence*100,2)), (box[0]+200, box[1]+30),
  #         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

  cv2.imshow("Output", frame)
  result.write(frame)
  if cv2.waitKey(1) & 0xff == ord('q'):
    break

result.release()