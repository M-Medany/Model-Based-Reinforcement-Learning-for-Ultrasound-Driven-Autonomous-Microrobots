import cv2

# q
    # sudo chmod a+rw /dev/ttyACM0

cap = cv2.VideoCapture(0)

#Skip the first 10 frames
for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame during initialization. Exiting.")
        break

if not cap.isOpened():
    print("Camera not accessible")

else:
    ret, frame = cap.read()
    
    if ret:
        while True:
            ret, frame = cap.read()
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    else:
        print("Failed to read frame. Exiting.")
        cap.release()
        cv2.destroyAllWindows()