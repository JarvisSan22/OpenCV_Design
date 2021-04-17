
import cv2


def ResizePixilate(img,factor): 
    """
    Risize image and keep the inter area, this will keep the pixles from sized down images when resized up.
    """
    h,w,c=img.shape
    img_small=cv2.resize(img,(w//factor,h//factor),200)
    img_pix=cv2.resize(img_small,(w,h),interpolation = cv2.INTER_AREA)
    return img_pix