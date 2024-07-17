# Import needed libraries(packages)
from ultralytics import YOLO
import cv2

"""
In this code you use the detection model on images
Enter your image path with its format in single or double quotation
Example = "my_image_path.jpg"
"""

# Introduce your pytorch model(.pt) to YOLO , in single or double quotation
model = YOLO()
results = model(source="",show=True)
cv2.waitKey()
cv2.destroyAllWindows()
