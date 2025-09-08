import numpy as np
import cv2
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
from utils.tracking_CSRT import CSRT_tracker
from utils.actuator import Arduino, FunctionGenerator_1
import yaml

cap = cv2.VideoCapture(0)

for _ in range(5):
    tre, frame = cap.read()
    #cv2.imshow('ROI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open(f'scripts/config.yaml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)



board=Arduino(config['Arduino_settings'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
print("Arduino initialized successfully")

function_generator = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
print("Function generator initialized successfully")

# while(True):
#     # Capture frame-by-frame
#     tre, frame = cap.read()
#     if tre:
#         print(frame.shape)
#         # print(frame

#         # Our operations on the frame come here
#         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         #roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

#         # Display the resulting frame
#         cv2.imshow('ROI', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         raise(ConnectionAbortedError("Could not connect to camera"))
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


if tre:
    roi_x, roi_y, roi_width, roi_height = cv2.selectROI("Draw Bounding Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

print(roi_x, roi_y, roi_width, roi_height)

tre, frame = cap.read()
cropped_resized, cropped = resize_and_crop_frame(frame, roi_x, roi_y, roi_width, roi_height)

binary_mask, bitmap = create_binary_bitmap(cropped, manual=True)
print(bitmap)
print(cropped)

print('segmentation completed successfully')
cv2.imshow('segmented image', (bitmap*255))
cv2.waitKey(0)
cv2.imshow('segmented image', (binary_mask*255))
cv2.waitKey(0)

cv2.destroyAllWindows()

final_image = plot_cluster_on_image_blue(bitmap, cropped, threshold=100)
print('cluster plotted successfully')

cv2.imshow('final image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

tracker = CSRT_tracker(False, initial_image=final_image)

tre, frame = cap.read()
frame_cleaned, _ = resize_and_crop_frame(frame, roi_x, roi_y, roi_width, roi_height)

frame_cleaned = plot_cluster_on_image_blue(bitmap, frame_cleaned, threshold=100)



success, bbox = tracker.tracker.update(frame_cleaned)
annotated_frame = np.copy(frame_cleaned)

if success:
    # Tracking successful, update the bounding box
    x, y, w, h = [int(i) for i in bbox]
    annotated_frame = cv2.rectangle(frame_cleaned, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Calculate center coordinates
    center_x = x + w // 2
    center_y = y + h // 2
    print("Center coordinates:", center_x, center_y)
    

else:
    # Tracking failed, you may want to handle this case
    raise Exception("Tracking failed at frame: ")

# Break the loop if 'q' is pressed



#cv2.imshow("CSRT Tracking", annotated_frame)
cv2.imshow("CSRT Tracking", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()







while(True):
    # Capture frame-by-frame
    tre, frame = cap.read()
    if tre:
        print(frame.shape)
        _, frame = resize_and_crop_frame(frame, roi_x, roi_y, roi_width, roi_height)
        cleaned_image = plot_cluster_on_image_blue(bitmap, frame)

        success, bbox = tracker.update(cleaned_image)
        annotated_frame = np.copy(cleaned_image)

        if success:
            # Tracking successful, update the bounding box
            x, y, w, h = [int(i) for i in bbox]
            annotated_frame = cv2.rectangle(cleaned_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate center coordinates
            agent_location = np.array([x + w // 2, y + h // 2], dtype=int)
            print('agent location', agent_location)
            

        else:
            # Tracking failed, we may want to handle this case
            raise Exception("Tracking failed; Exception at step: ") 


       
        cv2.circle(annotated_frame, tuple([0,0]), 5, (0, 0, 255), -1)
        cv2.imshow("Annotated frame", annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    










# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

