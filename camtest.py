import cv2

def test_camera_access(camera_index=0):
    # Try to open the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return False
    
    # Try to read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        return False
    
    print("Camera access successful. Frame captured.")
    cv2.imshow("Captured Frame", frame)
    cv2.waitKey(0)  # Display the frame for 2 seconds
    # Release the camera
    cap.release()
    return True 

if __name__ == "__main__":

    test_camera_access(1)

# import cv2
# for i in range(10):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"✅ Camera {i} 可用")
#         cap.release()
#     else:
#         print(f"❌ Camera {i} 不可用")