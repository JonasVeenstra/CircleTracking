import numpy as np
import matplotlib.pyplot as plt
import os,sys
import matplotlib.cm as cm
import pickle
import cv2


root = '/Users/Jonas/Documents/LocalData'
path = '/dec2023'
name  ='/test3'
ext = '.avi'
video_path = root+path+name+ext

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    print(1)
    ret, f = cap.read()
    if not ret:
        break
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    lower = np.array([40,40,40])
    upper = np.array([255,255,255])

    mask = cv2.inRange(f, lower, upper)
    masked = cv2.bitwise_and(f,f, mask=mask)

    result = f- masked
    
    
    plt.imshow(255-result)
    plt.show()
    # gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 
    # plt.imshow(gray)
    # plt.show()
    break

cap.release()
cv2.destroyAllWindows()



# 
# def filter_very_black(frame):
#     # Convert the frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     plt.imshow( gray_frame)
#     plt.show()
#     # Define a threshold to identify very black pixels
#     _, thresholded = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY)
# 
#     plt.imshow( thresholded)
#     plt.show()
# 
# 
# 
#     # Create a mask by inverting the thresholded image
#     mask = cv2.bitwise_not(thresholded)
# 
#     # Apply the mask to the original frame
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     plt.imshow( result)
#     plt.show()
# 
#     return result
# 
# # Open a video file
# # video_path = 'your_video_file.mp4'  # Replace with the path to your video file
# cap = cv2.VideoCapture(video_path)
# 
# while cap.isOpened():
#     print(1)
#     ret, frame = cap.read()
# 
#     if not ret:
#         break
# 
#     # Filter very black colors
#     filtered_frame = filter_very_black(frame)
# 
#     # Display the original and filtered frames
#     # plt.imshow(frame)
#     plt.imshow( filtered_frame)
#     plt.show()
#     # Break the loop if 'q' key is pressed
#     # if cv2.waitKey(25) & 0xFF == ord('q'):
#     break
# 
# cap.release()
# cv2.destroyAllWindows()
