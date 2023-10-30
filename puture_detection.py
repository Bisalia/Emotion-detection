# Importing OpenCV package
import cv2

def main():
    # Reading the image
    img = cv2.imread('assets/happy_cheldren.jpeg')
    
    # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
    
    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    
    # Iterating through rectangles of detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite('outputs/Detection_faces.jpg', img)

if __name__=='__main__':
    main()

