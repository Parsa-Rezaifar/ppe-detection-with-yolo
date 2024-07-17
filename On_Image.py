# Import needed libraries(packages)
from ultralytics import YOLO
import cv2

# Introduce your pytorch model(.pt) to YOLO
model = YOLO()
# Your image path
results = model(source="",show=True)
cv2.waitKey()
cv2.destroyAllWindows()