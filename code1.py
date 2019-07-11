import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


class opencvtest:
    #change color
    def light_color(self,img,num):
        B,G,R=cv2.split(img)
        temp=[]
        for x in [B,G,R]:
            if num == 0:
                pass
            elif num >0:
                lim = 255 - num
                x[x>lim] = 255
                x[x <= lim] = (num +x[x<=lim]).astype(img.dtype)
            else:
                lim =0 - num
            temp.append(x)
        img_merge = cv2.merge(tuple(temp))
        return img_merge
        # cv2.imshow('gakki_change', img_merge)
        # key = cv2.waitKey() 
        # if key == 27: 
        #     cv2.destroyAllWindows()
    #gama correction
    def gamma_correction(self,img,gamma):
        invGamma = 1.0/gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** invGamma) * 255)
        table = np.array(table).astype("uint8")
        img_brighter = cv2.LUT(img, table)
        return img_brighter
        # cv2.imshow('gakki_change', img_brighter)
        # key = cv2.waitKey() 
        # if key == 27: 
        #     cv2.destroyAllWindows()

    #scale and rotation
    def scaleandrotation(self,img,scale,rotation):
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotation, scale) # center, angle, scale
        img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img_rotate
        # cv2.imshow('gakki_change', img_rotate)
        # key = cv2.waitKey() 
        # if key == 27: 
        #     cv2.destroyAllWindows()

    #Affine Transform
    def affine_transform(self,img,pst):
        rows,cols,ch = img.shape
        pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])
        M = cv2.getAffineTransform(pts1,pts)
        img_dst = cv2.warpAffine(img, M, (cols, rows))
        return img_dst

    # perspective transform
    def perspective_transform(self,img,margin):
        height,width,chanels = img.shape
        x1 = random.randint(-margin, margin)
        y1 = random.randint(-margin, margin)
        x2 = random.randint(width - margin - 1, width - 1)
        y2 = random.randint(-margin, margin)
        x3 = random.randint(width - margin - 1, width - 1)
        y3 = random.randint(height - margin - 1, height - 1)
        x4 = random.randint(-margin, margin)
        y4 = random.randint(height - margin - 1, height - 1)

        dx1 = random.randint(-margin, margin)
        dy1 = random.randint(-margin, margin)
        dx2 = random.randint(width - margin - 1, width - 1)
        dy2 = random.randint(-margin, margin)
        dx3 = random.randint(width - margin - 1, width - 1)
        dy3 = random.randint(height - margin - 1, height - 1)
        dx4 = random.randint(-margin, margin)
        dy4 = random.randint(height - margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        return  img_warp


if __name__ =="__main__":
    #import img
    cvtest=opencvtest()
    img = cv2.imread('1.jpg')
    #img_change=cvtest.light_color(img,30)
    #img_change=cvtest.gamma_correction(img,1.5)
    #img_change=cvtest.scaleandrotation(img,0.5,45)
    #pts=np.float32([[100,100],[300,0],[0,500]])
    #img_change=cvtest.affine_transform(img,pts)
    img_change = cvtest.perspective_transform(img,90)
    cv2.imshow('gakki_change',img_change)
    key = cv2.waitKey() 
    if key == 27: 
        cv2.destroyAllWindows()

