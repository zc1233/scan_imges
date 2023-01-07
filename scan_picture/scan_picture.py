#扫描图片

import cv2
import numpy as np
import matplotlib.pyplot as plt

def cv_show(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)
    cv2.imwrite('./' + name + '.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#将四个角坐标排序：左上， 右上， 左下， 右下
def order_points(pts):
    sort_id = np.argsort(pts[:,0])
    pts = pts[sort_id, :]
    mid_sort_id = np.argsort(pts[0:2, 1])
    pts[0:2, :] = pts[mid_sort_id, :]
    mid_sort_id = np.argsort(pts[2:4, 1])
    mid_pts = pts[2:4, :]
    pts[2:4 :] = mid_pts[mid_sort_id, :]
    return pts

#计算两点间曼哈顿距离
def distance(x , y):
    dis = np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))
    return dis

#图像坐标变换
def four_point_transform(img , pts):
    rect = order_points(pts)
    (tl, tr, bl, br) = rect

    #计算轮廓参数
    widthA = distance(br, bl)
    widthB = distance(tr, tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heigthA = distance(tr, br)
    heigthB = distance(tl, bl)
    maxHeigth = max(int(heigthA), int(heigthB))
    
    #变换后坐标
    dst = np.array([
        [0, 0],
        [0, maxWidth-1],
        [maxHeigth-1, 0],
        [maxHeigth-1, maxWidth-1]
    ],dtype = np.float32)
    #数据类型转换
    rect = rect.astype(np.float32)
    
    #计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxHeigth, maxWidth))
    return warped

# 获取当前运行路径
path = os.getcwd()
# 获取所有文件名的列表
filename_list = os.listdir(path)
# 获取所有word文件名列表
imagename_list = [filename for filename in filename_list \
                    if filename.endswith((".jpg", ".png", ".jpeg"))]

now_threshold = input("请输入图像阈值0-255(扫描效果受此影响,一般为100):")
now_threshold = int(now_threshold)

print("转换中...")
for picture_name in imagename_list:
    picture_for_scan = cv2.imread(picture_name , 0)
    origina_picture = cv2.imread(picture_name , 1)
    color_picture = origina_picture.copy()
    #cv_show('origina_picture', origina_picture)

    #滤波
    blur_picture = cv2.GaussianBlur(picture_for_scan, (5, 5), 0)
    #cv_show('blur_picture', blur_picture)

    #canny算子，边缘检测（双阈值处理）
    #edged_picture = cv2.Canny(blur_picture, 50, 150)
    edged_picture = blur_picture.copy()
    ret , edged_picture = cv2.threshold(edged_picture, now_threshold, 255, cv2.THRESH_BINARY) 
    #cv_show('edged_picture', edged_picture)

    contours , hes = cv2.findContours(edged_picture.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours_picture = cv2.drawContours(color_picture, contours, -1, (0, 255, 0), 2)
    #cv_show('draw_contours_picture', draw_contours_picture)

    if len(contours) > 0:
        #按面积大小排序
        contours = sorted(contours, key = cv2.contourArea , reverse = True)
        
    c = contours[0]
    #draw_contours_picture = cv2.drawContours(color_picture, c, -1, (0, 255, 0), 2)
    #cv_show('draw_contours_picture', draw_contours_picture)
    peri = cv2.arcLength(c, True)
    for k in np.arange(0,0.2,0.01):
        approx = cv2.approxPolyDP(c, k*peri, True)
        if len(approx) == 4:
            docCnt = approx
            scaned_picture = four_point_transform(picture_for_scan.copy() , docCnt.reshape(4, 2))
            #scaned_picture = cv2.cvtColor(scaned_picture, cv2.COLOR_BGR2GRAY) 
            ret , scaned_picture = cv2.threshold(scaned_picture, 0, 255, cv2.THRESH_OTSU)
            scaned_picture_name = 'scaned_' + os.path.splitext(picture_name)[0]
            cv2.imwrite('./' + scaned_picture_name + '.png', scaned_picture)
            #cv_show(scaned_picture_name, scaned_picture)
            break

print("转换完成")
print("若不满意，请重新选择阈值或拍摄一张纸片轮廓清晰的图片")
os.system('pause')