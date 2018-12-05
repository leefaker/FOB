# encoding:utf-8
import cv2
import numpy as np
#得到模板函数
def get_t_black(f):
    if f==0:
        template = np.zeros((120, 350), dtype=np.uint8)
        template[1:120,1:350]=255
        template[0:120,80:81]=0
        template[0:120,120:121]=0

def get_template(height,width,top_wid,left_wid, num_tem):
    #num_tem==1代表黑卡，黑卡的模板如下
    if num_tem==1:
        template = np.zeros((height, width * 3), dtype=np.uint8)
        # template[top_wid:height-top_wid,left_wid:width-left_wid]=255
        # template[0:height, 5:11] = 0
        template[top_wid:height-top_wid,0:width*3]=255
        template[0:height,width:width+1]=0
        template[0:30, width+int(width/2):width + int(width/2)+1] = 0
        template[height-30:height,width+int(width/2):width + int(width/2)+1]=0
        template[0:height,width*2:width*2+1] = 0

    #白卡的模板如下
    if num_tem==0:
        template = np.zeros((height, width*2), dtype=np.uint8)
        # template[top_wid:height - top_wid, 18:23] = 255
        # template[0:30,20:21]=0
        # template[90:120, 20:21] = 0
        template[:,:]=255
        template[0:1 ,0:width]=0
        template[height-1:height, 0:width] = 0
        template[0:height,0:1]=0
        template[0:height,width:width+1]=0
    #距离变换

    dis = cv2.distanceTransform(template, distanceType=cv2.DIST_L1, maskSize=3, dstType=cv2.CV_8U)
    #cv2.normalize(dis, dis)
    di = np.zeros((dis.shape[0],dis.shape[1]), dtype=np.uint8)
    #格式转换由32位转换为八位无符号
    # for i in range(dis.shape[0]):
    #     for j in range(dis.shape[1]):
    #         num = dis[i][j]
    #         di[i][j] = int(num)
    #cv2.imshow("d",di)
    #返回距离变换后的图像
    return dis
#得到FOB卡区域的函数
def get_ori(image):
    #规定模板的长度为40，高度为125，实际上模板的宽度是具体情况而定
    height=125
    width=40
    #cv2.imshow('template',template)
    #大致分割
    ori=image[300:660,100:950]
    image=image[300:660,100:950]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #转灰度图
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #取左上角一小块区域灰度和判断是白卡还是黑卡
    sum_gray=np.sum(image[0:20,2:4])
    #中值滤波
    image = cv2.medianBlur(image, 19)
    #得到模板加canny计算边缘
    if sum_gray < 60*20*2:#flag=1 黑卡 flag=0白卡
        flag=1
        img = cv2.Canny(image, 1, 20)
        template_0 = get_template(height, width, 1, 1, 1)
    else:
        flag=0
        template_0 = get_template(height, width, 1, 1, 0)
        img = cv2.Canny(image, 1,40)

    # x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    # y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    # absX = cv2.convertScaleAbs(x)
    # #转回uint8
    # absY = cv2.convertScaleAbs(y)
    # dst = cv2.addWeighted(absX,0.8,absY,0.2,0)
    # _,binary=cv2.threshold(dst,0,255,cv2.THRESH_OTSU)
    # binary=~binary
    # cv2.imshow("dsd",binary)
    #由于距离变换是白点对黑点计算距离 所以要取反
    img=~img
    # wid=10
    # for i in range(img.shape[0]):
    #     j=0
    #     while j<img.shape[1]:
    #         if 0 in img[i,j:j+wid]:
    #             img[i,j:j+wid]=255
    #             img[i][j+int(wid/2)]=0
    #         j = j + wid
    #cv2.imshow('Canny',img)
    dis=cv2.distanceTransform(img,distanceType=cv2.DIST_L1,maskSize=3,dstType=cv2.CV_8U)
    #cv2.normalize(dis,dis)
    #cv2.imshow("dis",dis*255)
    #距离变换后转换为八位无符号整形
    # di=np.zeros(image.shape,dtype=np.uint8)
    # for i in range(dis.shape[0]):
    #     for j in range(dis.shape[1]):
    #         num=dis[i][j]
    #         if num>255:
    #             num=255
    #         di[i][j]=int(num)
    #模板匹配选择的函数
    method=cv2.TM_SQDIFF
    #模板匹配
    res= cv2.matchTemplate(dis,template_0,method)
    #找到模板匹配函数最小值和所在的位置
    min_val_0, max_val_0, min_loc_0, max_loc_0 = cv2.minMaxLoc(res)
    #如果最小值小于某一阈值那么出现了高度匹配的
    if min_loc_0[0] - 400>0:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res[:, min_loc_0[0] - 400:min_loc_0[0] - 150])
        min_loc=(min_loc[0]+min_loc_0[0] - 400,min_loc[1])
        left= min_loc_0[0]-300
        if min_loc_0[0]-min_loc[0]>300:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res[:, left:min_loc_0[0] - 150])
            min_loc = (min_loc[0] + left, min_loc[1])
    else:
        min_val, max_val, min_loc, max_loc=min_val_0, max_val_0, min_loc_0, max_loc_0
    if flag==0:
        #print("出现了高匹配度的")
        # for i in range(min_loc[0],min_loc[0]+20):
        #     if 0 in img[min_loc[1]-5:min_loc[1]+5,i:i+1]:
        #         num=num+1
        #     if 0 in img[min_loc[1] + 55:min_loc[1] + 65, i:i + 1]:
        #         num=num+1
        #
        # if num >40:
        #     g=1
        #如果最小函数处坐标左边还出现了比较高度匹配的 则选取左边的点
        num=0
        a=[]
        b=[]
        score=0
        if min_loc[0]+80>850 and min_loc[1]+100>360:
            top_left=min_loc_0
        else:
            for i in range(min_loc[0]+10,min_loc[0]+70):
                a.append(np.mean(gray[min_loc[1]+50:min_loc[1]+70,i:i+1]))
            min_=min(a[0:60])
            max_=max(a[0:60])
            for i in range(len(a)):
                b.append(70*(a[i]-min_)/(max_-min_))
            max_decay = 0
            pos=30
            for i in range(27,33):
                if np.mean(a[i:60])-np.mean(a[0:i])>max_decay:
                    max_decay=np.mean(a[i:60])-np.mean(a[0:i])
                    pos=i
            for i in range(60):
                if i<pos:
                    score=score+b[i]
                else:
                    score=score+70-b[i]
            score=score-max(a[pos:60])+min(a[0:pos])
            # for i in range(min_loc_0[0]+10,min_loc_0[0]+70):
            #     a.append(np.mean(gray[min_loc_0[1]+50:min_loc_0[1]+70,i:i+1]))
            # min_=min(a[0:60])
            # max_=max(a[0:60])
            # for i in range(len(a)):
            #     b.append(20*(a[i]-min_)/(max_-min_))
            # max_decay=0
            # for i in range(1,59):
            #     if np.mean(a[i:60])-np.mean(a[0:i])>max_decay:
            #         max_decay=np.mean(a[i:60])-np.mean(a[0:i])
            #         pos=i
            # for i in range(60):
            #     if i<pos:
            #         score=score+b[i]
            #     else:
            #         score=score+20-b[i]
            gg=np.mean(a[5:pos])
            if  (score<1200 and gg<100) or score<800:#min_val<500000 and (min_loc_0[0]-min_loc[0]<200 or min_val_0-min_val<100000):
                top_left=min_loc
            else:
                top_left=min_loc_0
    else:
        #在不越界的情况下
        if min_loc_0[0]+102<850 and min_loc_0[1]+78<360 :
            right_boundry = 840
            for i in range(min_loc_0[0]+100,850):
                if np.min(gray[min_loc_0[1]+50:min_loc_0[1]+70,i:i+1])<30 and np.max(gray[min_loc_0[1]+50:min_loc_0[1]+70,i:i+1])<100:
                    right_boundry=i
                    break
            #如果右边界距离标签坐标距离小于250代表匹配到了第二条条纹此时应该取左边条纹
            if right_boundry-min_loc_0[0]<270 and right_boundry-min_loc_0[0]>0:
                top_left = min_loc
            else:
                top_left = min_loc_0
        #if  min_val<min_val_0+250000 and (min_loc_0[0]-min_loc[0]<200 or min_val_0-min_val<100000):
        else:
            top_left=min_loc_0
        # for i in range(0,res.shape[1]):
        #     res[:,i]=res[:,i]+i*i*1.5
    #min_val_0, max_val_0, min_loc_0, max_loc_0 = cv2.minMaxLoc(res)

    #由于黑卡和白卡选取模板长度不一样，所以加入下面操作
    if flag==1:
        top_left=(top_left[0] + width, top_left[1] )
    # else:
    #     top_left = (top_left[0] , top_left[1])
    bottom_right = (top_left[0] + width, top_left[1] + height)
    # cv2.rectangle(di,top_left, bottom_right, 255, 2)
    # cv2.rectangle(ori,top_left, bottom_right, 255, 2)
    mid=(top_left[0]+20,top_left[1])
    if mid[0]-100<0:
        mid=(100,mid[1])
    else:
        if mid[0]+250>ori.shape[1]:
            mid=(ori.shape[1]-252,mid[1])
    if mid[1]+125>ori.shape[0]:
        mid = (mid[0],ori.shape[0]-130)
    img_ori=ori[mid[1]:mid[1]+125,mid[0]-100:mid[0]+250]
    #print("wwee")
    return img_ori
# image=cv2.imread('bug2.bmp')
# ori=get_ori(image)
# #print(image)
# cv2.imshow('ori',ori)
# cv2.waitKey(0)

# s="dadad"
# print(s.split('.')[-1])