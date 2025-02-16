# https://docs.ultralytics.com/modes/track/#python-examples
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
# model = YOLO(r'weights\3_classes_weight\weights\best.pt').to("cuda") 
model = YOLO(r'').to("cuda") 
count = 0
obj_id_list = []
all_id_list = []
obj_id_list_pingbi = []
track_list =[]
counting_buffer= {}


w_ratio_left=0.3
w_ratio_right=0.6
h_ratio_up= 0.2
h_ratio_down= 0.8



# Open the camera
# cap = cv2.VideoCapture(0)
# video_path = r'E:\git_hub\yolov8_tracking_3_classes_newyolo\videos\Downward_looking_video\2\D01_20230723170724.mp4'
video_path = r''
outvideo_path = r''


cap = cv2.VideoCapture(video_path)

# Get Camera Parameter
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
fps = cap.get(cv2.CAP_PROP_FPS)  
frame_size = (int(width), int(height))  

# 初始化 VideoWriter
out = cv2.VideoWriter(outvideo_path, fourcc, fps, frame_size)


def calculate_point(w, h, w_ratio_left, w_ratio_right, h_ratio_up, h_ratio_down):
    # point1 = (int(w_ratio_left * w), int(h - h_ratio_down * h))
    # point2 = (int(w_ratio_right * w), int(h - h_ratio_down * h))
    # point3 = (int(w_ratio_left * w), int(h - h_ratio_up * h))
    # point4 = (int(w_ratio_right * w), int(h - h_ratio_up * h))
    point1 = (int(w_ratio_left * w), int(h - h_ratio_up * h))
    point2 = (int(w_ratio_right * w), int(h - h_ratio_up * h))
    point3 = (int(w_ratio_right * w), int(h - h_ratio_down * h))
    point4 = (int(w_ratio_left * w), int(h - h_ratio_down * h))
    return point1, point2, point3, point4


def draw_rectangle(im0,point1, point2, point3, point4, color=(0, 0, 0)):
    # color() = (0, 0, 0)
    cv2.line(im0, point1, point2, color, thickness=3)
    cv2.line(im0, point2, point3, color, thickness=3)
    cv2.line(im0, point3, point4, color, thickness=3)
    cv2.line(im0, point4, point1, color, thickness=3)
    # region_points = [point1, point2, point3, point4]


def is_inside_region(point, region_points):
    
    return cv2.pointPolygonTest(np.array(region_points), point, False) >= 0


def count_objects_in_region(boxes, region_points, id, cls):

    count = 0
    for box in boxes:
        x_center, y_center, width, height = box
        center_point = (int(x_center), int(y_center))
        if is_inside_region(center_point, region_points):
            count += 1
    return count




def count_objects_in_region(boxes, region_points, detect_id, detect_cls):
    count = 0
    objects_in_region = []  
    for obj_id, obj_cls, box in zip(detect_id, detect_cls, boxes):
        x1, y1, x2, y2 = box  
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        center_point = (int(x_center), int(y_center))

        if is_inside_region(center_point, region_points):
            count += 1
            objects_in_region.append({'id': obj_id, 'class': int(obj_cls)})  
    return count, objects_in_region


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)  
        # results = model.track(frame, tracker="bytetrack", persist=True, verbose=False)
        # model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Get Data for counting
        result = results[0].cpu().boxes
        detect_id = result.id.tolist() if result.id != None else []
        detect_cls = result.cls.tolist() if result.cls != None else []
        detect_xyxy = result.xyxy.tolist() if result.xyxy != None else []

        frame_counting_buffer = dict(zip(detect_id, zip(detect_cls, detect_xyxy))) # 关联数据
        
        
        h, w = annotated_frame.shape[:2]
        point1, point2, point3, point4 = calculate_point(w, h, w_ratio_left, w_ratio_right, h_ratio_up, h_ratio_down)
        region_points = [point1, point2, point3, point4]
        draw_rectangle(annotated_frame, point1, point2, point3, point4)


        
        # count, objects_in_region = count_objects_in_region(detect_xyxy, region_points, detect_id, detect_cls)

        
        for obj_id, (cls, box) in frame_counting_buffer.items():
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            center_point = (int(x_center), int(y_center))

            if is_inside_region(center_point, [point1, point2, point3, point4]):
                counting_buffer[obj_id] = (cls, center_point)

                
                if obj_id not in all_id_list:
                    all_id_list.append(obj_id)  



 

        for obj_id, (cls, _) in counting_buffer.items():
            info_text = f"ID: {obj_id}, Class: {cls}"
            cv2.putText(annotated_frame, info_text, (10, int(50 + obj_id * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        # Display the annotated frame
        cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Tracking", 1920, 1080)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        out.write(annotated_frame)  


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

