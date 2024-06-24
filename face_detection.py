import cv2

face_capture = cv2.CascadeClassifier("C:/Users/SHAIKH ADI/AppData/Roaming/Python/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml")

if face_capture.empty():
    print("Error loading cascade classifier!")
    exit()

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error opening video capture!")
    exit()

while True:
   
    ret, video_data = video_capture.read()
    
    if not ret:
        print("Failed to capture image")
        break


    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    

    faces = face_capture.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

  
    cv2.imshow("video_live", video_data)
    
  
    if cv2.waitKey(10) == ord("a"):
        break


video_capture.release()
cv2.destroyAllWindows()
