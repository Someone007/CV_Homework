import numpy as np 
import cv2

def medianblur(img,kernel,padding_way='ZERO'):
    if kernel % 2 == 0 or kernel == 1:
        return None
    #计算paddingsize
    paddingsize = kernel //2
    channel=len(img.shape)
    height,width = img.shape[0:2]
    
    if channel == 3:
        matout = np.zeros_like(img)
        for x in range(matout.shape[2]):
            matout[:,:,x] =   medianblur(img[:,:,x],kernel,padding_way)
        return matout
    elif channel ==2:
        matbase = np.zeros((height+paddingsize*2,width+paddingsize*2),dtype=img.dtype)
        matbase[paddingsize:-paddingsize,paddingsize:-paddingsize] = img
        if padding_way=='ZERO':
            pass
        elif padding_way=='REPLICA':
            for x in range(paddingsize):
                matbase[x,paddingsize:paddingsize+height]=img[0,:]
                matbase[-(x+1),paddingsize:paddingsize+height]=img[-1,:]
                matbase[paddingsize:paddingsize+width,x]=img[:,0]
                matbase[paddingsize:paddingsize+width,-(x+1)]=img[:,-1]

            matout = np.zeros((height, width), dtype=img.dtype)
            for i in range(height):
                for j in range(width):
                    line = matbase[i:i+kernel,j:j+kernel].flatten()
                    line = np.sort(line)
                    matout[i,j] = line[(kernel*kernel) // 2]
            return matout
        else:
            print("padding_way error need ZERO or REPLICA")
            return None

if __name__ =="__main__":
    img=cv2.imread('D://CV/img/11.jpg')
    img_mdb=medianblur(img,5,padding_way='REPLICA')
    if img_mdb is None:
        print("None")

    # 调用OpenCV的接口进行中值滤波
    opencv = cv2.medianBlur(img, 5)

    # 这里进行图片合并
    img = np.hstack((img, img_mdb))
    img = np.hstack((img, opencv))

    # 显示对比效果
    cv2.imshow('ORG + myself + OpenCV', img)

    cv2.waitKey()
    cv2.destroyAllWindows()
