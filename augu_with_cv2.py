'''
- use of cv2 library for image augmentation
- Rotate, Horizontal and vertical flip, Increase brightness
'''

import cv2
import os
import glob
import sys
import argparse
import numpy as np
#from IPI_ML.data_operations import count 

def resize(image, width = 256, height = 256):
    res_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return res_img

def resize_images(image, height=256, width=256):
    actual_height, actual_width = image.shape[:2]
    max_height, max_width =  height, width
    if max_height < actual_height or max_width < actual_width:
        scaling_factor = max_height/float(actual_height)
        if max_width/float(actual_width) < scaling_factor:
            scaling_factor = max_width /float(actual_width)
        resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return resized_image

def rote(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def vertical_flip(image):
        vertical = cv2.flip(image, 1)
        return vertical

def horizontal_flip(image):
        horizontal = cv2.flip(image, 0)
        return horizontal

def increase_brightness(image,bright_intensity):
  bright = np.ones(image.shape,dtype='uint8')*bright_intensity
  bright_increase = cv2.add(image,bright)
  return bright_increase


def main(args):

    arg = argparse.ArgumentParser()

    arg.add_argument(
        "--resize",
        default = False,
        help = "Resizes images to desired size. Square is better. Default value is 28x28" 
    )
    arg.add_argument(
        "--rotate",
        default = True,
        help = "Setting bool value for rotation. Set 'True' if required. "
    )
    arg.add_argument(
        "--horizontal_flip",
        default = False,
        help = "Setting bool value for horizontal flip, Set 'True' if required. "
    )
    arg.add_argument(
        "--vertical_flip",
        default = True,
        help = "Setting bool value for vertical flip, Set 'True' if required."
    )
    arg.add_argument(
        "--increase_brightness",
        default = False,
        help = "Setting bool value for increasing brightness, Set 'True' if required."
    )
    
    args = arg.parse_args()
    counter =1
    
    path_test_images='D:\BV_4cam\CP1 Data_Deleted Multiple Copies/aug'
    out_path = 'D:\BV_4cam\CP1 Data_Deleted Multiple Copies/aug_output'
    
    for dir_, _, files in os.walk(path_test_images):
        relDir = os.path.relpath(dir_, path_test_images)
        no_test_imgs = (len(files))
        total_count= len(files)
        print(no_test_imgs)
        for name in files:
            file = (dir_ + "/"+str(name))
            print(file)
            image =  cv2.imread(file,0)
            shape = image.shape[:2]
            if args.resize:
                res = resize(image, width=256,height=256)
                cv2.imwrite(args.output_path+file, res)
            if args.rotate:
                angles = np.arange(15,360,15)
                # angles = [30,270]
                for angle in angles:
                    rotated = rote(image, angle)
                    pos = file.rfind('/')
                    cv2.imwrite(out_path+relDir+"/"+'rot'+str(angle)+"_"+str(name),rotated)
                    #cv2.imwrite(output_path+'rot'+str(angle)+str(file[pos+1:]),rotated)
            if args.horizontal_flip:
                hor = horizontal_flip(image)
                pos = file.rfind('/')
                cv2.imwrite(out_path+relDir+"/"+'hor'+str(name),hor)
                #cv2.imwrite(output_path+'hor'+str(file[pos+1:]),hor)
            if args.vertical_flip:
                ver = vertical_flip(image)            
                pos = file.rfind('/')
                cv2.imwrite(out_path+relDir+"/"+'ver'+str(name), ver)
            if args.increase_brightness:
                img = increase_brightness(image,bright_intensity=50)
                pos = file.rfind('/')
                cv2.imwrite(out_path+relDir+"/"+'bright'+str(name), img)

            sys.stdout.write("\rTotal images processed : %d " % counter)
            sys.stdout.flush()

            while counter <= total_count:
                counter += 1
                break
            #except:
            #    print("Image corrupted name = {}".format(file))
        print('\n')
    
if __name__=="__main__":
    main(sys.argv)
