# Import needed libraries(packages)
from ultralytics import YOLO
import cvzone
import math
import cv2

# Reading frames from webcam using cv2.VideoCapture(0)
"""
if you want to use any video sources , use cv2.VideoCapture("Your video path.format")
Note : Enter your video format like : .mp4,...
Example : capture = cv2.VideoCapture("Your video path.mp4")
"""
capture = cv2.VideoCapture(0)
# Resize captured video
capture.set(propId=3,value=1280)
capture.set(propId=4,value=720)

# Introduce your pytorch model(.pt) to YOLO
your_model = YOLO('PPE_Detection_Model.pt')

# All objects that we can detect with this model , including : 
# Personal Protective Equipments(PPEs)
# Construction machines
# Industrial machines
# General vehicles
# Fire related
# Work tools
# General
class_names = [
    'Excavator','Gloves','Hardhat','Ladder','Mask','NO-Hardhat',
    'NO-Mask','NO-Safety Vest','Person','SUV','Safety Cone',
    'Safety Vest','bus','dump truck','fire hydrant','machinery','mini-van',
    'sedan','semi','trailer','truck and trailer','truck','van',
    'vehicle','wheel loader'
]

# Default color to fill in with our specified colors
my_color = (None,None,None)

# Read
while True :
    ret , frame = capture.read()
    results = your_model(frame,stream=True)
    for result in results :
        boxes = result.boxes
        # Create bounding boxes
        for box in boxes :
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            width,height = x2-x1,y2-y1
            # Calculate confidence
            confidence = math.ceil((box.conf[0]*100)) / 100
            # Access class names
            cls = int(box.cls[0])
            current_class = class_names[cls]
            # Detect ppe based on specific confidence
            # 0.5 means 50 percent
            if confidence > 0.5 : 
                if current_class == 'NO-Hardhat' or current_class == 'NO-Safety Vest' or current_class == 'NO-Mask' :
                    my_color = (0,0,255)
                elif current_class == 'Hardhat' or current_class == 'Safety Vest' or current_class == 'Mask' or current_class == 'Safety Cone' or current_class == 'fire hydrant' or current_class == 'Gloves':
                    my_color = (0,255,0)
                elif current_class == 'Person' :
                    my_color = (255,0,0)
                else :
                    my_color = (0,0,0)
            cvzone.putTextRect(frame,text=f'{class_names[cls]}-{confidence}',pos=(max(0,x1),max(35,y1)),
                               scale = 2,
                               thickness = 2,
                               colorB = my_color,
                               colorT = (255,255,255),
                               colorR = my_color,
                               offset = 5)
            cv2.rectangle(frame,(x1,y1),(x2,y2),my_color,3)
    # Show the result
    cv2.imshow('PPE detection system',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27 :
        break