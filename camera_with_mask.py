import cv2
import numpy as np
def put_mask(img, dog_mask):
    dog_mask_rgb = dog_mask[:, :, :3]  # Utilize only the RGB channels of the mask
    dog_mask_h, dog_mask_w, _ = dog_mask_rgb.shape
    img_h, img_w, _ = img.shape
    # Resize the mask to fit on face
    factor = min(img_h / dog_mask_h, img_w / dog_mask_w)
    new_mask_w = int(factor * dog_mask_w)
    new_mask_h = int(factor * dog_mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)

    # Add mask to face - ensure mask is centered
    resized_mask = cv2.resize(dog_mask_rgb, new_mask_shape)
    
    img_with_mask = img.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    off_h = int((img_h - new_mask_h) / 2)  
    off_w = int((img_w - new_mask_w) / 2)
    img_with_mask[off_h: off_h + new_mask_h, off_w: off_w + new_mask_w][non_white_pixels] = \
         resized_mask[non_white_pixels]
    return img_with_mask
def main():
    cap = cv2.VideoCapture(0)

    dog_mask = cv2.imread('assets/dog.png')
    # initialize front face classifier
    cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

    while True:
        re, df = cap.read()
        df_h, df_w, _ = df.shape
        # Convert to black-and-white
        gray_cap = cv2.cvtColor(df, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray_cap)

        # Detect faces
        rects = cascade.detectMultiScale(
            blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # Add all bounding boxes to the image
        for x, y, w, h in rects:
           # crop a frame slightly larger than the face
            y0, y1 = int(y - 0.25*h), int(y + 0.75*h)
            x0, x1 = x, x + w
        # give up if the cropped frame would be out-of-bounds
            if x0 < 0 or y0 < 0 or x1 > df_w or y1 > df_h:
                continue

            # apply mask
            df[y0: y1, x0: x1] = put_mask(df[y0: y1, x0: x1], dog_mask)

        # Display the resulting frame
        cv2.imshow('frame', df)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    
if __name__=='__main__':
    main()