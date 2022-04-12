'''
k-means（k均值聚类算法，无监督学习）
将数据分为k组，然后随机选取k个点作为聚类中心，遍历每一个点，计算这个点和
每一个聚类中心的距离，将这个点分配到离它最近的那个组中，遍历完之后对每一组
重新计算聚类中心，循环直到聚类中心不再改变，或者每一个点所属类别不再改变
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

#数据量
DATA_NUM=1000
#数据尺度：默认二维
DATA_SCALE=2
#数据值范围
DATA_LIMIT=[-1000,1000]
#数据组数
DATA_GROUP=8
#聚类中心
DATA_CENTER=[]
#图片分辨率(900*900)
IMG_SIZE=9
#帧序号
FRAME_NUM=0
#文件路径
ROOT='./result'
#是否要合成视频
TO_VIDEO=True
#迭代次数（若聚类中心不变则结束迭代）
EPOCH=50
#延迟帧数（聚类中心不变后继续获取的帧数）
DELTA_FRAME=5

#生成数据
def Generate_data():
    dataArr=np.random.randint(low=DATA_LIMIT[0],high=DATA_LIMIT[1],size=(DATA_NUM,DATA_SCALE))
    dataGroup=np.random.randint(low=0,high=DATA_GROUP,size=(DATA_NUM,1))
    #往narray插入一列数据
    dataArr=np.append(dataArr,dataGroup,axis=1)
    return dataArr

#绘散点图函数：colormap有
def Draw_plot(dataArr,colormap='jet'):
    global FRAME_NUM,DATA_CENTER
    #代表图片大小：900*900
    plt.figure(figsize=(IMG_SIZE,IMG_SIZE),edgecolor='gray')
    plt.tight_layout()
    plt.scatter(dataArr[:,0],dataArr[:,1],c=dataArr[:,2],cmap=colormap,alpha=1,s=75,marker='.')
    temp_center=np.array(DATA_CENTER)
    plt.scatter(temp_center[:,0],temp_center[:,1],c='r',s=3000,alpha=0.3,marker='o')
    plt.savefig(ROOT+'/imgs/'+str(FRAME_NUM)+'.png')
    plt.close()
    FRAME_NUM+=1

#随机挑选聚类中心
def Select_center(dataArr):
    global DATA_CENTER
    center=np.random.choice(DATA_NUM,DATA_GROUP,replace=False)
    for c in center:
        DATA_CENTER.append([dataArr[c][0],dataArr[c][1]])

#更新数据所属组别
def Update_group(data):
    global DATA_CENTER
    distance=[]
    for center in DATA_CENTER:
        distance.append(np.sqrt(pow(center[0]-data[0],2)+pow(center[1]-data[1],2)))
    min_distance_index=distance.index(min(distance))
    return min_distance_index

#更新聚类中心位置
def Update_center(dataArr):
    global DATA_CENTER
    sum_X=np.zeros(DATA_GROUP)
    sum_Y=np.zeros(DATA_GROUP)
    num=np.zeros(DATA_GROUP)
    for data in dataArr:
        sum_X[data[2]]+=data[0]
        sum_Y[data[2]]+=data[1]
        num[data[2]]+=1
    sum_X=[round(x/y,1) for x,y in zip(sum_X,num)]
    sum_Y=[round(x/y,1) for x,y in zip(sum_Y,num)]
    DATA_CENTER=[[x,y] for x,y in zip(sum_X,sum_Y)]

#转换成视频
def To_vedio():
    import cv2,os,time
    t=time.localtime()
    nowtime=str(t.tm_year)[-2:]+str(t.tm_mon)+str(t.tm_mday)+str(np.random.randint(100))
    #指定编码器
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoname=ROOT+'/'+nowtime+'k_means.avi'
    #文件名、编码器、帧率（1s多少张）、分辨率
    videoWriter=cv2.VideoWriter(videoname, fourcc, 1, (IMG_SIZE*100, IMG_SIZE*100))
    filenames=os.listdir(ROOT+'/imgs')
    for file in filenames:
        img=cv2.imread(ROOT+'/imgs/'+file)
        videoWriter.write(img)
    videoWriter.release()
    print("--视频合成完毕！--")

def Clean_imgs():
    import os
    filename=os.listdir(ROOT+'/imgs')
    for file in filename:
        os.remove(ROOT+'/imgs/'+file)


if __name__=='__main__':
    #清空imgs文件夹
    Clean_imgs()
    dataArr=Generate_data()
    Select_center(dataArr)
    for epoch in range(EPOCH):
        Draw_plot(dataArr)
        for i in range(DATA_NUM):
            new_GPnum = Update_group(dataArr[i])
            dataArr[i][2] = new_GPnum
        old_centers=DATA_CENTER
        Update_center(dataArr)
        print(epoch)
        if old_centers==DATA_CENTER:
            DELTA_FRAME-=1
            print("True")
            if DELTA_FRAME==0:break
    if TO_VIDEO:
        To_vedio()





