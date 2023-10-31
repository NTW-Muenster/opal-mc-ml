from PIL import Image, ImageOps, ImageChops
# Lade das PNG-Bild
import os
import math
import skimage
from skimage import io, transform, color
import numpy as np
import cv2

def distance_line_point(line_point1, line_point2, point):
    x1, y1 = line_point1
    x2, y2 = line_point2
    x, y = point
    # Berechnung des Abstands
    distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return distance

def distance_between_points(x1, y1, x2, y2):
    # Calculate the distance between two points (x1, y1) and (x2, y2)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - (w * zoom), y - (h * zoom), 
                    x + (w * zoom), y + (h * zoom)))
    return img.resize((w, h))

directory = 'events-images'
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        imgname=filename
        image = Image.open('events-images/'+imgname)
        rgb_image = image.convert('RGB')
        hsv_image = image.convert('HSV')
        # Konvertiere das Bild in eine Numpy-Array-Matrix
        pixel_matrix = hsv_image.load()

        #for i in range(0,200):
            #pixel_matrix[0,i]=(0,0,0,0)
        #Mitte and aeusserstem Kreis finden
        kord2x = 9999
        kreiskoords = np.array([])
        for y in range(99,100):
            for x in range(2, 198):
                pixel = pixel_matrix[x, y]
                h, s, v = hsv_image.getpixel((x, y))
                if (s==0 and v!=0):
                    kreiskoords = np.append(kreiskoords, int(x))
                    kord1x=x
                    kord1y=y
                    if (x<kord2x):
                        kord2x=x
                        kord2y=y
                    
        print(kreiskoords)
        #print(kord1x, kord1y, 'UND', kord2x, kord2y)
        
        kreiskoords2 = np.array([])
        kord4x = 9999
        for q in range(2,199):
            for p in range(99, 100):
                pixel = pixel_matrix[p, q]
                h, s, v = hsv_image.getpixel((p, q))
                if (s==0 and v!=0):
                    kreiskoords2 = np.append(kreiskoords2, int(q))
                    kord3x=p
                    kord3y=q
                    if (p<kord4x):
                        kord4x=p
                        kord4y=q
        #print(kord3x, kord3y, 'UND', kord4x, kord4y)
        middlex=(kord1x+kord2x)/2
        #print('middlex', int(middlex))
        middley=(kord3y+kord4y)/2
        #print('middley', int(middley))
        #FAILCASE HANDLING
        if ((int(abs(99.5-middlex))>6) or (int(abs(99.5-middley))>6)):
            image.save('centered-masked-rotated/exception/'+imgname)
            i=5
            while (int(abs(99.5-middlex))>6):
                kord2x = 9999
                for yy in range(60+i,61+i):
                    for xx in range(2, 198):
                        pixel = pixel_matrix[xx, yy]
                        h, s, v = hsv_image.getpixel((xx, yy))
                        if (s==0 and v!=0):
                            kord1x=xx
                            kord1y=yy
                            if (x<kord2x):
                                kord2x=xx
                                kord2y=yy
                #print(kord3x, kord3y, 'UND', kord4x, kord4y)
                middlex=int((kord1x+kord2x)/2)
                #print('middlex', int(middlex))
                
                i+=5
            j=5
            while (int(abs(99.5-middley))>6):
                kord4x = 9999
                for qq in range(2,199):
                    for pp in range(60+j, 61+j):
                        pixel = pixel_matrix[pp, qq]
                        h, s, v = hsv_image.getpixel((pp, qq))
                        if (s<10 and v!=0):
                            kord3x=pp
                            kord3y=qq
                            if (p<kord4x):
                                kord4x=pp
                                kord4y=qq
                middley=int((kord3y+kord4y)/2)
                #print('middley', int(middley))
                j+=5
        
        else:
            if (abs((kord1x+kord2x)-153)<4):
                rescale=(abs(kord1x-kord2x)*1.0)/153
            else:
                rescale=(abs(kord1x-kord2x)*1.0)/99
                
        arr=np.add(kreiskoords,-int(middlex))
        arr2=np.add(kreiskoords2,-int(middley))
        arr2=abs(arr2)
        arr=np.abs(arr)
        arrplus=np.add(arr,1)
        arrplus2=np.add(arr2,1)
        arrminus=np.add(arr,-1)
        arrminus2=np.add(arr2,-1)
        arrminus=np.abs(arrminus)
        arrminus2=np.abs(arrminus2)
        arr=np.append(arr,arrplus)
        arr=np.append(arr,arrminus)
        arr=np.append(arr,arr2)
        arr=np.append(arr,arrplus2)
        arr=np.append(arr,arrminus2)
        arr=np.unique(arr)
        print(arr,imgname)
        
        for y in range(0,200):
                for x in range(0,200):
                        point = (x, y)
                        distance = distance_between_points(middlex, middley, x , y)
                        if (np.any((arr == int(distance)))):
                            h, s, v = hsv_image.getpixel((x, y))
                            if (s<10):
                                image.putpixel( (x, y), (0, 0, 0, 255) ) 
                            
        rgb_image_masked  = image.convert('RGB')
        hsv_image_masked  = image.convert('HSV')
        # Konvertiere das Bild in eine Numpy-Array-Matrix
        pixel_matrix_masked  = hsv_image_masked .load()
                  
        minimaldistance=1000000000000000000
        for winkel in range(0,181):
            steigung=math.tan(math.radians(winkel+.1))
            line_point1 = (middlex, middley)
            line_point2 = (middlex+10, 10*steigung + middley)
            sumdist=0.0
            #print('sumdist before', sumdist)
            for y in range(2,197):
                for x in range(2,197):
                    pixel = pixel_matrix_masked[x, y]
                    r, g, b = rgb_image_masked.getpixel((x, y))
                    if (r!=0 or g!=0 or b!=0):
                        point = (x, y)
                        dist = distance_line_point(line_point1, line_point2, point)
                        sumdist += dist
                        #print('sumdist', sumdist)
                        
            if (sumdist<minimaldistance):
                minimaldistance=sumdist
                minimalwinkel=winkel
            #print('sumdist after', sumdist)
            #print('angle after', 90.0-minimalwinkel)

            #print(filename, sumdist, 90+winkel,90+minimalwinkel) 
            #print("Abstand:", distance)
        #print(filename, sumdist, 90-minimalwinkel)            
            
        
            
            
        
            #Bild verschieben
        offset=ImageChops.offset(image, int(99.5-middlex), int(99.5-middley))
        offset.save('centered-masked-rotated/centered-masked/centered-masked'+imgname)
        file2 = cv2.imread('centered-masked-rotated/centered-masked/centered-masked'+imgname)
        
        


        # Get the image size
        (height, width) = file2.shape[:2]

        # Define the center of the image
        center = (width / 2, height / 2)

        # Perform the general rotation
        angle =90+minimalwinkel
        scale = 1.0
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rot3 = cv2.warpAffine(file2, matrix, (width, height))
        #file = os.path.join('centeredrotated2', 'centered'+imgname)
        
        #rot=offset.rotate(90+minimalwinkel)
        #image_skimage = io.imread(file)
        #rot2=skimage.transform.rotate(image_skimage,90+minimalwinkel)
            #offset.show()
            #Bild abspeichern
        cv2.imwrite('centered-masked-rotated/centeredmaskedrotatedopencv'+imgname, rot3)
        #rot.save('centeredrotated2/centeredrotated'+imgname)
        #io.imsave('centeredrotated2/centeredrotated2'+imgname , rot2)
        
directory2 = 'centered-masked-rotated'
for filename in os.listdir(directory2):
    if filename.endswith('.png'):
        imgname=filename
        image = Image.open('centered-masked-rotated/'+imgname)
        rgb_image = image.convert('RGB')
        hsv_image = image.convert('HSV')
        # Konvertiere das Bild in eine Numpy-Array-Matrix
        pixel_matrix = hsv_image.load()
        lower=0
        for y in range(100,200):
            for x in range(0, 200):
                pixel = pixel_matrix[x, y]
                h, s, v = hsv_image.getpixel((x, y))
                if (v!=0):
                    lower=lower+1
        upper=0
        for y in range(0,100):
            for x in range(0, 200):
                pixel = pixel_matrix[x, y]
                h, s, v = hsv_image.getpixel((x, y))
                if (v!=0):
                    upper=upper+1
                    
        #image.putpixel( (25, 25), (255, 255, 255, 255) )
        if(lower>upper):
            print("I was flipped", "lower",lower,"upper",upper, imgname)
            flip=image.transpose(Image.FLIP_TOP_BOTTOM)
            flip.save('centered-masked-rotated/final/final'+imgname)
        else:
            print("I was not flipped", "lower",lower,"upper",upper, imgname)
            image.save('centered-masked-rotated/final/final'+imgname)



