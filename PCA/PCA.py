'''
PCA算法步骤
设有 m 条 n 维数据。
1、将原始数据按列组成 n 行 m 列矩阵 X；
2、将 X 的每一行进行零均值化，即减去这一行的均值；求出协方差矩阵 C=(X*X.T)/m ；
3、求出协方差矩阵的特征值及对应的特征向量；
4、将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 P；Y=PX 即为降维到 k 维后的数据。
'''
import cv2,os
import numpy as np
'''人脸识别中的默认分类器,主要用于人脸图片中的人脸轮廓的识别'''
FaceDetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#源数据保存路径
DIRECTORY_SRC='./general_data'
#灰度图保存路径
DIRECTORY_GRAY='./facedata_gray'
#脸部图片的尺寸
FACE_SIZE=80
#降维后的特征数
T_FEATURE_NUM=40

#转灰度图像并裁剪图像
def Crop_face(filename,imgname):
    img=cv2.imread(filename)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        facezone=FaceDetector.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5,minSize=(100,100))
        for x,y,w,h in facezone:
            print(imgname)
            #裁剪：保留脸部图片
            img=img[y:y+h,x:x+w]
            if img is not None:
                #缩放：统一尺寸
                img=cv2.resize(img,(FACE_SIZE,FACE_SIZE),interpolation=cv2.INTER_CUBIC)
                if imgname !='test.jpg':
                    cv2.imwrite(DIRECTORY_GRAY+'/'+imgname,img)
                else:
                    return img
                # cv2.imshow('face',img)
                # cv2.waitKey()

#去中心化
def Decentration(data):
    mean=np.around(np.mean(data,axis=0))
    newdata=np.empty(shape=(0,FACE_SIZE*FACE_SIZE))
    for linedata in data:
        linedata=np.array([x-y for x,y in zip(linedata,mean)])
        newdata=np.append(newdata,linedata.reshape(1,FACE_SIZE*FACE_SIZE),axis=0)
    return newdata

#求协方差矩阵,返回特征向量矩阵P与经变换后的矩阵result
def Covariance_matrix(data):
    C_matrix=np.round(np.dot(data.T,data)/data.shape[0])
    value,vector=np.linalg.eig(C_matrix)
    P=vector[:,0:T_FEATURE_NUM]
    result=np.round(np.dot(data,P))
    return P,result

#求目标向量和其他向量的距离
def Min_distance(srcdata,target):
    distance=[]
    for data in srcdata:
        d=[pow(x-y,2) for x,y in zip(data,target)]
        d=sum(d)
        distance.append(d)
    return distance.index(min(distance))

if __name__=='__main__':
    '''预处理数据'''
    # data=os.listdir(DIRECTORY_SRC)
    # for img in data:
    #     Crop_face(DIRECTORY_SRC+'/'+img,img)

    '''PCA'''
    # data=os.listdir(DIRECTORY_GRAY)
    # features=np.empty(shape=(0,FACE_SIZE*FACE_SIZE))
    # for filename in data:
    #     img=cv2.imread(DIRECTORY_GRAY+'/'+filename)
    #     '''抛去部分尺寸异常数据'''
    #     if img.shape !=(FACE_SIZE,FACE_SIZE,3):continue
    #     feature=img[:,:,0]
    #     feature=feature.ravel().reshape(1,FACE_SIZE*FACE_SIZE)
    #     features=np.append(features,feature,axis=0)
    # '''去中心化'''
    # data=Decentration(features)
    # '''计算协方差矩阵，求特征值与特征向量并返回特征向量矩阵、经变换后的矩阵'''
    # P,C_data=Covariance_matrix(data)
    # P=np.save('P.npy',P)
    # C_data=np.save('Changed_data.npy',C_data)
    '''测试'''
    P=np.load('P.npy')
    Changed_data=np.load('Changed_data.npy')
    target=Crop_face('./test.jpg','test.jpg')
    target=target.ravel().reshape(1,FACE_SIZE*FACE_SIZE)
    target=np.dot(target,P).reshape(T_FEATURE_NUM)
    best_index = Min_distance(Changed_data, target)
    print("最匹配人脸序列：",best_index)
    pass

