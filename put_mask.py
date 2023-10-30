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
    img = cv2.imread('assets/petite.png')
    dog_mask = cv2.imread('assets/dog.png', cv2.IMREAD_UNCHANGED)
    img_with_mask = put_mask(img, dog_mask)
    cv2.imwrite('outputs/image_with_dog_mask.png', img_with_mask)

if __name__ == '__main__':
    main()
