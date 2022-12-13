import cv2
def readtest():
    # 320527帧
    videoname = '/home/wangfz/wksp/EI/001001.avi'
    capture = cv2.VideoCapture(videoname)
    if capture.isOpened():
        count = 1
        while True:
            ret,img=capture.read() # img 就是一帧图片
            if not ret:
                break # 当获取完最后一帧就结束
            # if count%30==1:  
                # 
            cv2.imwrite(f'/home/wangfz/datasets/my_video/images/001001_{count}.png',img)
            print(count)
            count = count+1   
            if count == 1000:
                break 
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            # if not ret:
            #     break # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')

readtest()