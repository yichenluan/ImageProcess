#-*- coding: utf-8 -*-
from Tkinter import*
import os
from PIL import Image,ImageTk,ImageDraw
import tkFileDialog
from tkMessageBox import*
import cv
import cv2
import numpy as np
from math import *

filepath=''
filepath_pre=''

class Node:
    def __init__(self, value = None, pix = None, huffAns = '', parent = None, left = None, right = None):
        self.value = value
        self.pix = pix
        self.huffAns = huffAns
        self.parent = parent
        self.left = left
        self.right = right 
def huffman():
    imgHuff = img.convert('L')
    imgPix = imgHuff.load()
    allPix = width * height
    pixChance = [0 for i in range(256)]
    dicPix = {}
    huffmanTree = list()
    for i in range(width):
        for j in range(height):
            pixChance[imgPix[i,j]] += 1
    for i in range(256):
        if pixChance[i] != 0 :
            pixChance[i] = float(pixChance[i]) / allPix
            dicPix[i] = Node(pixChance[i], i, '')
            huffmanTree.append(dicPix[i])
    while (len(huffmanTree) >1):
        minNum = 99
        minK = None
        minNode = None
        lessMinNode = None
        for items in huffmanTree:
            if items.value <= minNum:
                minNum = items.value
                minNode = items
        huffmanTree.remove(minNode)
        minNum = 99
        for items in huffmanTree:
            if items.value <= minNum:
                minNum = items.value
                lessMinNode = items
        huffmanTree.remove(lessMinNode)
        newNode = Node(minNode.value + lessMinNode.value, left = minNode, right = lessMinNode)
        minNode.huffAns = '0'
        minNode.parent = newNode
        lessMinNode.huffAns = '1'
        lessMinNode.parent = newNode
        huffmanTree.append(newNode)
    for k,v in dicPix.items():
        tempNode = dicPix[k].parent
        while tempNode.parent :
            dicPix[k].huffAns = tempNode.huffAns + dicPix[k].huffAns
            tempNode = tempNode.parent
    print('*********************************')
    for k,v in dicPix.items():
        print k,    v.huffAns
    print('*********************************')

def fano():
    imgFano = img.convert('L')
    imgPix = imgFano.load()
    allPix = width * height
    pixChance = [0 for i in range(256)]
    dicPix = {}
    fanoTree = list()
    for i in range(width):
        for j in range(height):
            pixChance[imgPix[i,j]] += 1
    for i in range(256):
        if pixChance[i] != 0 :
            pixChance[i] = float(pixChance[i]) / allPix
            dicPix[i] = Node(pixChance[i], i, '')
            fanoTree.append((pixChance[i], dicPix[i]))
    fanoTree.sort()
    parentNode = Node()
    def buildTree(parentNode,fanoTree,fanoAns):
        if len(fanoTree) == 1:
            fanoTree[0][1].parent = parentNode
            fanoTree[0][1].huffAns = ''
            return parentNode
        else :
            valueAll = 0
            valueNow = 0
            for i in range(len(fanoTree)):
                valueAll += fanoTree[i][0]
            for i in range(len(fanoTree)):
                valueNow += fanoTree[i][0]
                if valueNow > float(valueAll) / 2:
                    break
            leftNode = Node(parent = parentNode, huffAns = 0)
            rightNode = Node(parent = parentNode, huffAns = 1)
            leftTree = fanoTree[:i]
            rightTree = fanoTree[i:]
            parentNode.left = buildTree(leftNode,leftTree,0)
            parentNode.right = buildTree(rightNode,rightTree,1)
    parentNode = buildTree(parentNode, fanoTree, 0)
    for k,v in dicPix.items():
        tempNode = dicPix[k].parent
        while tempNode.parent :
            dicPix[k].huffAns = str(tempNode.huffAns) + dicPix[k].huffAns
            tempNode = tempNode.parent
    print('------------------------------------')
    for k,v in dicPix.items():
        print k,    v.huffAns
    print('------------------------------------')
         
        
        




def choose_pic():
    '''选择图像函数
    
    根据路径选择一个具体的图像'''
    global filepath
    global filepath_pre
    global img
    global width
    global height
    filepath=tkFileDialog.askopenfilename()
    if filepath:
        filepath_pre=filepath
        img=Image.open(filepath)
        width,height=img.size
        show_pic(img)
        show_pic_process(img)
        show_hist(img)
        show_hist_process(img)


def show_pic(img):
    '''显示图像函数

    显示选择的未经处理的图像'''
    im=ImageTk.PhotoImage(img)
    cv_pri.create_image((256,256),image=im)
    cv_pri.im=im
    
def show_pic_process(img):
    '''显示处理后图像函数

    '''
    global im_process
    im_process=img
    img_process=ImageTk.PhotoImage(im_process)
    cv_process.create_image((256,256),image=img_process)
    cv_process.img_process=img_process


def saved():
    global img
    filepath_saved='/home/warlock/Desktop/temp'
    os.rename(filepath,filepath_saved)
    imgProcess.save(filepath)
    os.remove(filepath_saved)
    show_pic(imgProcess)
    show_hist(imgProcess)
    img = Image.open(filepath)
    


def show_hist(im):
    '''显示原图灰度直方图函数

    将会在图像下方显示灰度直方图和一些具体信息'''
    img_show=hist_process(im)
    img=ImageTk.PhotoImage(img_show)
    cv_hist_pri.create_image((256,50),image=img)
    cv_hist_pri.img=img
    st='总像素: '+str(pix_all)+'  '+'平均灰度: '+str(grey_avg)+'   '+'中值灰度: '+str(grey_mid)+'  '+'标准差: '+str(pix_dev)[0:5]
    Lab_info_pri.configure(text=st)
    Lab_info_pri.text=st
    

def show_hist_process(im):
    '''显示处理后图像灰度直方图函数

    在图像下方显示灰度直方图和一些具体信息'''
    img_show=hist_process(im)
    img=ImageTk.PhotoImage(img_show)
    cv_hist_process.create_image((256,50),image=img)
    cv_hist_process.img=img
    st='总像素: '+str(pix_all)+'    '+'平均灰度: '+str(grey_avg)+'  '+'中值灰度: '+str(grey_mid)+'  '+'标准差: '+str(pix_dev)[0:5]
    Lab_info_process.configure(text=st)
    Lab_info_process.st=st


def save_pic():
    '''另保为函数

    在弹出框中选择保存路径和格式'''
    save_filepath=tkFileDialog.asksaveasfilename(defaultextension='.jpg',filetypes=([('JPG','*.jpg'),('PNG','*.png'),('GIF','*.gif')]))
    if save_filepath:
        im_process.save(save_filepath)


##
#图像增强
#
#包括: convolution, smoothen, sharpen, c_convolute, convolute  

def convolution(pix):
    '''计算卷积函数


    根据给定的模板数组，得到卷积后的图像'''
    global img_con
    pix_pre = img.load()
    for i in range(width):
        for j in range(height):
            if i in [0,width-1] or j in [0,height-1]:
                 pix[i,j] = pix_pre[i,j]
            else :
                a_r = [0]*9
                a_g = [0]*9
                a_b = [0]*9
                for k in range(3):
                    for l in range(3):
                        a_r[k*3+l] = pix_pre[i-1+k,j-1+l][0]
                        a_g[k*3+l] = pix_pre[i-1+k,j-1+l][1]
                        a_b[k*3+l] = pix_pre[i-1+k,j-1+l][2]
                sum_r = 0
                sum_g = 0
                sum_b = 0
                for m in range(9):
                    sum_r = sum_r+ mod[m]*a_r[m]
                    sum_g = sum_g+ mod[m]*a_g[m]
                    sum_b = sum_b+ mod[m]*a_b[m]
                pix[i,j] = (int(sum_r), int(sum_g), int(sum_b))

def l_convolute(pix):
    global img_l
    img_pre = img.convert('L')
    pix_pre = img_pre.load()
    for i in range(width):
        for j in range(height):
            if i in [0,width-1] or j in [0,height-1]:
                pix[i,j] = pix_pre[i,j]
            else :
                a_pix = [0]*9
                for k in range(3):
                    for l in range(3):
                        a_pix[k*3+l] = pix_pre[i-1+k,j-1+l]
                sum_pix = 0
                for m in range(9):
                    sum_pix = sum_pix + mod[m]*a_pix[m]
                pix[i,j] = int(sum_pix)




def smoothen():
    '''图像平滑


    从Menu中得到模板数组，传递给卷积函数，显示平滑后图像'''
    global imgProcess
    global mod
    img_smoo = Image.new('RGB',img.size)
    pix_smoothen = img_smoo.load()
    model = int(vSmooth.get()[-1])
    mod = form_smooth[model-1] 
    convolution(pix_smoothen)
    show_pic_process(img_smoo)
    show_hist_process(img_smoo)
    imgProcess = img_smoo


def sharpen():
    '''图像锐化

    从Menu中得到模板数组，传递给卷积函数，显示锐化后图像'''
    global imgProcess
    global mod
    img_sharpen = Image.new('RGB',img.size)
    pix_sharpen = img_sharpen.load()
    model = int(vSharp.get()[-1])
    mod = form_sharp[model-1]
    convolution(pix_sharpen)
    show_pic_process(img_sharpen)
    show_hist_process(img_sharpen)
    imgProcess = img_sharpen

def c_convolute():
    '''用户自定义模板
    弹出对话框，自由填写模板数组值
    将数组传递给convolute函数'''
    global value_matrix
    value_matrix = StringVar()
    c_convolute = Toplevel()
    Label(c_convolute, text='请顺序输入矩阵的值:', width = 40).grid(row = 0,column = 0)
    Entry(c_convolute, textvariable = value_matrix).grid(row = 1,column = 0)
    Button(c_convolute, text ='确定', command = convolute).grid(row = 2,column = 0)


def convolute ():
    '''自定义模板卷积函数

    由c_convolute得到卷积模板，计算并显示'''
    global imgProcess
    global mod
    img_con = Image.new('RGB',img.size)
    pix_con = img_con.load()
    array = value_matrix.get().split()
    mod = [int(num) for num in array]
    convolution(pix_con)
    show_pic_process(img_con)
    show_hist_process(img_con)
    imgProcess = img_con
     

def edge_laplace():
    global mod
    global imgProcess
    img_laplace = Image.new('RGB',img.size)
    pix_laplace = img_laplace.load()
    mod = [1, 1, 1, 1, -8, 1, 1, 1 ,1]
    convolution(pix_laplace)
    show_pic_process(img_laplace)
    show_hist_process(img_laplace)
    imgProcess = img_laplace
    

def edge_kirsch():
    global mod
    global imgProcess
    img_kirsch = Image.new('L',img.size)
    pix_kirsch = pix_kirsch_1 = pix_kirsch_2 = pix_kirsch_3 = pix_kirsch_4 = pix_kirsch_5 = pix_kirsch_6 = pix_kirsch_7 = pix_kirsch_8 = img_kirsch.load()
    mod1 = [5, 5, 5, -3, 0, -3, -3, -3, -3]
    mod2 = [-3, 5, 5, -3, 0, 5, -3, -3, -3]
    mod3 = [-3, -3, 5, -3, 0, 5, -3, -3, 5]
    mod4 = [-3, -3, -3, -3, 0, 5, -3, 5, 5]
    mod5 = [-3, -3, -3, -3, 0, -3, 5, 5, 5]
    mod6 = [-3, -3, -3, 5, 0, -3, 5, 5, -3]
    mod7 = [5, -3, -3, 5, 0, -3, 5, -3, -3]
    mod8 = [5, 5, -3, 5, 0, -3, -3, -3, -3]
    mod = mod1
    l_convolute(pix_kirsch_1)
    mod = mod2
    l_convolute(pix_kirsch_2)
    mod = mod3
    l_convolute(pix_kirsch_3)
    mod = mod4
    l_convolute(pix_kirsch_4)
    mod = mod5
    l_convolute(pix_kirsch_5)
    mod = mod6
    l_convolute(pix_kirsch_6)
    mod = mod7
    l_convolute(pix_kirsch_7)
    mod = mod8
    l_convolute(pix_kirsch_8)
    for i in range (width):
        for j in range(height):
            pix_kirsch[i,j]=max(pix_kirsch_1[i,j], pix_kirsch_2[i,j], pix_kirsch_3[i,j], pix_kirsch_4[i,j], pix_kirsch_5[i,j], pix_kirsch_6[i,j], pix_kirsch_7[i,j], pix_kirsch_8[i,j])
    show_pic_process(img_kirsch)
    show_hist_process(img_kirsch)
    imgProcess = img_kirsch
    



def convert_L():
    '''灰度化函数
    根据原图的R，G，B分量得到灰度值
    显示灰度后图像'''
    global imgProcess
    if filepath_pre=='':
        showwarning(title='Error!',
                    message='Please Open a Picture!'
                    )
    else:
        img_convert=Image.new('L',img.size)
        pix_pre=img.load()
        pix_convert=img_convert.load()
        for i in range(width):
            for j in range(height):
                pix_convert[i,j]=pix_pre[i,j][0]*0.299+pix_pre[i,j][1]*0.587+pix_pre[i,j][2]*0.114
        show_pic_process(img_convert)
        show_hist_process(img_convert)
        imgProcess = img_convert

##
#傅立叶变换和离散余弦变换
#
#包括: FFT, FImage, four, lisanyuxian
def FFT(image,flag=0):
    w = image.width
    h = image.height
    iTmp = cv.CreateImage((w,h),cv.IPL_DEPTH_32F,1)
    cv.Convert(image,iTmp)
    iMat=cv.CreateMat(h,w,cv.CV_32FC2)
    mFFT=cv.CreateMat(h,w,cv.CV_32FC2)
    for i in range(h):
        for j in range(w):
            if flag == 0:
                num = -1 if (i+j)%2 ==1 else 1
            else :
                num = 1
            iMat[i,j] = (iTmp[i,j]*num,0)
    cv.DFT(iMat,mFFT,cv.CV_DXT_FORWARD)
    return mFFT

def FImage(mat):
    w = mat.cols
    h = mat.rows
    size = (w,h)
    iAdd = cv.CreateImage(size,cv.IPL_DEPTH_8U,1)
    for i in range(h):
        for j in range(w):
            iAdd[i,j] = mat [i,j][1]/h + mat[i,j][0]/h
    return iAdd
    
def four( ):
    image = cv.LoadImage(filepath,0)
    mAfterFFT = FFT(image)
    mBeginFFT = FFT(image,1)
    iAfter = FImage(mAfterFFT)
    iBegin = FImage(mBeginFFT)
    cv.ShowImage('iAfter',iAfter)
    cv.ShowImage('iBegin',iBegin)
    cv.WaitKey(0)
    
     

def lisanyuxian():
    '''离散余弦变换函数

    调用opencv库函数，进行离散余弦变换'''
    img1 = cv2.imread(filepath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    h,w = img1.shape[:2]
    vis0 = np.zeros((h,w),np.float32)
    vis0[:h,:w] = img1
    vis1 = cv2.dct(vis0)
    img2 = cv.CreateMat(vis1.shape[0],vis1.shape[1],cv.CV_32FC3)
    cv.CvtColor(cv.fromarray(vis1),img2,cv.CV_GRAY2BGR)
    print(img2)
    cv.ShowImage('离散余弦变换',img2)
    cv.WaitKey(0)

def canny():
    img1 = cv.LoadImage(filepath,0)
    PCannyImg = cv.CreateImage(cv.GetSize(img1), cv.IPL_DEPTH_8U,1)
    cv.Canny(img1,PCannyImg,50,150,3)
    cv.NamedWindow('canny',1)
    cv.ShowImage('canny',PCannyImg)
    cv.WaitKey(0)




def hist_equ():
    '''均衡化函数


    将灰度图均衡化后显示'''
    global imgProcess
    img_equ=Image.new('L',img.size)
    img_L=img.convert('L')
    pix_equ=img_equ.load()
    pix_img_L=img_L.load()
    hist=[0 for i in range(256)]
    pix_new=0
    for i in range(width):
        for j in range(height):
            hist[pix_img_L[i,j]]+=1
    for i in range(width):
        for j in range(height):
            for k in range(pix_img_L[i,j]+1):
                pix_new+=hist[k]
            pix_equ[i,j]=int(255*(float((pix_new-hist[0]))/(width*height-hist[0])))
            pix_new=0
    show_pic_process(img_equ)
    show_hist_process(img_equ)
    imgProcess = img_equ
                

    

def thin():
    global imgProcess
    dicPix = {}
    imgThin = img.convert('L')
    pixThin = imgThin.load()
    pixDelete = list()
    loop = True
    while (loop):
        for i in range(1,width-1):
            for j in range(1,height-1):
                if pixThin[i,j] == 0:
                    dicPix['p1'] = pixThin[i,j]
                    dicPix['p2'] = pixThin[i,j-1]
                    dicPix['p3'] = pixThin[i-1,j-1]
                    dicPix['p4'] = pixThin[i-1,j]
                    dicPix['p5'] = pixThin[i-1,j+1]
                    dicPix['p6'] = pixThin[i,j+1]
                    dicPix['p7'] = pixThin[i+1,j+1]
                    dicPix['p8'] = pixThin[i+1,j]
                    dicPix['p9'] = pixThin[i+1,j-1]
                    n = dicPix['p2']+dicPix['p3']+dicPix['p4']+dicPix['p5']+dicPix['p6']+dicPix['p7']+dicPix['p8']+dicPix['p9']
                    n = n/255
                    n = 8-n
                    if n >= 2 and n <= 6 :
                        z = 0
                        if dicPix['p2']- dicPix['p3'] == 255 :
                            z += 1
                        if dicPix['p3']- dicPix['p4'] == 255 :
                            z += 1
                        if dicPix['p4']- dicPix['p5'] == 255 :
                            z += 1
                        if dicPix['p5']- dicPix['p6'] == 255 :
                            z += 1
                        if dicPix['p6']- dicPix['p7'] == 255 :
                            z += 1
                        if dicPix['p7']- dicPix['p8'] == 255 :
                            z += 1
                        if dicPix['p8']- dicPix['p9'] == 255 :
                            z += 1
                        if dicPix['p9']- dicPix['p2'] == 255 :
                            z += 1
                        if z == 1 :
                            z = 0
                            if dicPix['p2']*dicPix['p8']*dicPix['p6'] ==0 and dicPix['p4']*dicPix['p8']*dicPix['p6'] ==0 :
                                pixDelete.append((i,j))
        if len(pixDelete) == 0:
            loop = False

        for (i,j) in pixDelete:
            pixThin[i,j] = 255
        pixDelete = pixDelete[:0]
        for i in range(1,width-1):
            for j in range(1,height-1):
                if pixThin[i,j] == 0:
                    dicPix['p1'] = pixThin[i,j]
                    dicPix['p2'] = pixThin[i,j-1]
                    dicPix['p3'] = pixThin[i-1,j-1]
                    dicPix['p4'] = pixThin[i-1,j]
                    dicPix['p5'] = pixThin[i-1,j+1]
                    dicPix['p6'] = pixThin[i,j+1]
                    dicPix['p7'] = pixThin[i+1,j+1]
                    dicPix['p8'] = pixThin[i+1,j]
                    dicPix['p9'] = pixThin[i+1,j-1]
                    n = dicPix['p2']+dicPix['p3']+dicPix['p4']+dicPix['p5']+dicPix['p6']+dicPix['p7']+dicPix['p8']+dicPix['p9']
                    n = n/255
                    n = 8-n
                    if n >= 2 and n <= 6 :
                        z = 0
                        if dicPix['p2']- dicPix['p3'] == 255 :
                            z += 1
                        if dicPix['p3']- dicPix['p4'] == 255 :
                            z += 1
                        if dicPix['p4']- dicPix['p5'] == 255 :
                            z += 1
                        if dicPix['p5']- dicPix['p6'] == 255 :
                            z += 1
                        if dicPix['p6']- dicPix['p7'] == 255 :
                            z += 1
                        if dicPix['p7']- dicPix['p8'] == 255 :
                            z += 1
                        if dicPix['p8']- dicPix['p9'] == 255 :
                            z += 1
                        if dicPix['p9']- dicPix['p2'] == 255 :
                            z += 1
                        if z == 1 :
                            z = 0
                            if dicPix['p2']*dicPix['p4']*dicPix['p6'] ==0 and dicPix['p4']*dicPix['p8']*dicPix['p2'] ==0 :
                                pixDelete.append((i,j))
        if len(pixDelete) != 0:
            loop = True
        for (i,j) in pixDelete:
            pixThin[i,j] = 255
        pixDelete = pixDelete[:0]

    imgProcess = imgThin
    show_pic_process(imgThin)
    show_hist_process(imgThin)


def circleCheck():
    global imgProcess
    imgPre = img.convert('L')
    imgPix = imgPre.load()
    dicLevel = {}
    dicUplevel = {}
    dicLevelCircle = {}
    dicUplevelCircle = {}
    for i in range(width):
        for j in range(height):
            if imgPix[i,j] >= 128 :
                imgPix[i,j] = 255
            else :
                imgPix[i,j] = 0
                if dicLevel.has_key(j):
                    dicLevel[j].append(i)
                else :
                    dicLevel[j] = [i]
                if dicUplevel.has_key(i):
                    dicUplevel[i].append(j)
                else :
                    dicUplevel[i] = [j]
    for j in range(height):
        if j not in dicLevel.keys():
            continue
        for itemA in dicLevel[j]:
            itemList = dicLevel[j]
            for itemB in itemList :
                if itemA == itemB:
                    continue
                middle = int((itemA +itemB)/2)
                if dicLevelCircle.has_key(middle):
                    dicLevelCircle[middle] += 1
                else :
                    dicLevelCircle[middle] = 1
            itemList.remove(itemA)
        dicLevel[j] = dicLevel[j][:0]
    for i in range(width):
        if i not in dicUplevel.keys():
            continue
        for itemA in dicUplevel[i]:
            itemList = dicUplevel[i]
            for itemB in itemList :
                if itemA == itemB:
                    continue
                middle = int((itemA + itemB)/2)
                if dicUplevelCircle.has_key(middle):
                    dicUplevelCircle[middle] += 1
                else :
                    dicUplevelCircle[middle] = 1
            itemList.remove(itemA)
        dicUplevel[i] = dicUplevel[i][:0]
    circleLevelA = 0
    circleUplevelA = 0
    circleLevelB = 0
    circleUplevelB =0
    maxNum = 0
    for keys in dicLevelCircle.keys():
        if dicLevelCircle[keys] >= maxNum:
            maxNum = dicLevelCircle[keys]
            circleLevelA = keys
    del dicLevelCircle[circleLevelA]
    maxNum =0
    for keys in dicLevelCircle.keys():
        if dicLevelCircle[keys] >= maxNum:
            maxNum = dicLevelCircle[keys]
            circleLevelB = keys
    maxNum = 0
    for keys in dicUplevelCircle.keys():
        if dicUplevelCircle[keys] >= maxNum:
            maxNum = dicUplevelCircle[keys]
            circleUplevelA = keys
    del dicUplevelCircle[circleUplevelA]
    maxNum = 0
    for keys in dicUplevelCircle.keys():
        if dicUplevelCircle[keys] >= maxNum:
            maxNum = dicUplevelCircle[keys]
            circleUplevelB = keys

    
    circleA = list()
    circleB = list()
    for i in range(width):
        for j in range(height):
            if imgPix[i,j]!= 0:
                continue
            ra = int(sqrt((i - circleLevelA)**2 + (j - circleUplevelA)**2))
            rb = int(sqrt((i - circleLevelB)**2 + (j - circleUplevelB)**2))
            circleA.append(ra)
            circleB.append(rb)
    dicCircleA = dict([(circleA.count(i),i) for i in circleA])
    ra = dicCircleA[max(dicCircleA.keys())] 
    dicCircleB = dict([(circleB.count(i),i) for i in circleB])
    rb = dicCircleB[max(dicCircleB.keys())] 
    circleDraw = ImageDraw.Draw(imgPre)
    circleDraw.line(((circleLevelA,circleUplevelA+ra),(circleLevelA,circleUplevelA+10)),fill = 0)
    circleDraw.line(((circleLevelA,circleUplevelA-10),(circleLevelA,circleUplevelA-ra)),fill = 0)
    circleDraw.line(((circleLevelA-ra,circleUplevelA),(circleLevelA-10,circleUplevelA)),fill = 0)
    circleDraw.line(((circleLevelA+10,circleUplevelA),(circleLevelA+ra,circleUplevelA)),fill = 0)
    circleDraw.line(((circleLevelA+5,circleUplevelA),(circleLevelA-5,circleUplevelA)),fill = 0)
    circleDraw.line(((circleLevelA,circleUplevelA+5),(circleLevelA,circleUplevelA-5)),fill = 0)

    circleDraw.line(((circleLevelB,circleUplevelB+rb),(circleLevelB,circleUplevelB+10)),fill = 0)
    circleDraw.line(((circleLevelB,circleUplevelB-10),(circleLevelB,circleUplevelB-rb)),fill = 0)
    circleDraw.line(((circleLevelB-rb,circleUplevelB),(circleLevelB-10,circleUplevelB)),fill = 0)
    circleDraw.line(((circleLevelB+10,circleUplevelB),(circleLevelB+rb,circleUplevelB)),fill = 0)
    circleDraw.line(((circleLevelB+5,circleUplevelB),(circleLevelB-5,circleUplevelB)),fill = 0)
    circleDraw.line(((circleLevelB,circleUplevelB+5),(circleLevelB,circleUplevelB-5)),fill = 0)
    print('----------------------------------------------------------------------')
    print('圆1：a = '+str(circleLevelA)+' b = '+str(circleUplevelA)+' r = '+str(ra))
    print('圆2：a = '+str(circleLevelB)+' b = '+str(circleUplevelB)+' r = '+str(rb))
    print('----------------------------------------------------------------------')

    show_pic_process(imgPre)
    show_hist_process(imgPre)
    imgProcess = imgPre
            

   
    


                    


                    


def lineChoose():
    global valueLine
    valueLine=StringVar()
    cLine=Toplevel()
    Label(cLine,text='choose',width=30).grid(row=0,column=0)
    sca=Scale(cLine,from_= 0,to=200,orient=HORIZONTAL,variable=valueLine).grid(row=1,column=0)
    Button(cLine,text='ok',command=houghLine).grid(row=2,column=0)

def houghLine():
    global mod
    global imgProcess
    threshold = int(valueLine.get())
    imgPre = img.convert('L')
    imgPix = imgPre.load()
    imgLine = img.convert('L')
    pixLine = imgLine.load()
    PI = pi
    radianSplit = 300
    radianMax = 2*PI
    radian = 0
    radius = 0
    dic = {}
    lineList = list()
    for i in range(width):
        for j in range(height):
            if imgPix[i,j] >= 128 :
                imgPix[i,j] = 255
            else :
                imgPix[i,j] = 0
    for i in range(width):
        for j in range(height):
            if imgPix[i,j] == 0 :
                for k in range(radianSplit+1):
                    radian = radianMax * (float(k)/radianSplit)
                    radius = int(i * cos(radian) + j * sin(radian))
                    if radius > 0 and radius < width-1 :
                        strHough = str(radian)+','+str(radius)
                        if dic.has_key(strHough) :
                            dic[strHough] += 1
                        else :
                            dic[strHough] = 1
    for (i,j) in dic.items():
        if j >= threshold:
            lineList.append((i,j))
    for (i,j) in lineList:
        ans = i.split(',')
        radian = float(ans[0])
        radius = int(ans[1])
        point1_x = int(radius/cos(radian))
        point1_y = 0
        point2_x = int((radius - (height - 1) * sin(radian))/cos(radian))
        point2_y = height-1
        drawLine = ImageDraw.Draw(imgLine)
        drawLine.line(((point1_x,point1_y),(point2_x,point2_y)),fill = 0)
    show_pic_process(imgLine)
    show_hist_process(imgLine)
    imgProcess = imgLine



    
    



def linear_en():
    global imgProcess
    multiple=float(vLinear_enhance.get())
    img_l_en=img.point(lambda i: i*multiple)
    show_pic_process(img_l_en)
    show_hist_process(img_l_en)
    imgProcess = img_l_en



def linear_fa():
    global imgProcess
    multiple=float(vLinear_fade.get())
    img_l_fa=img.point(lambda i:i*multiple)
    show_pic_process(img_l_fa)
    show_hist_process(img_l_fa)
    imgProcess = img_l_fa


def nlinear_en():
    global imgProcess
    multiple=float(vnLinear_enhance.get())
    img_nl_en=img.point(lambda i:(i+i*multiple*(255-i)/255))
    show_pic_process(img_nl_en)
    show_hist_process(img_nl_en)
    imgProcess = img_nl_en


def nlinear_fa():
    global imgProcess
    multiple=float(vnLinear_fade.get())
    img_nl_fa=img.point(lambda i:(i+i*multiple*(255-i)/255))
    show_pic_process(img_nl_fa)
    show_hist_process(img_nl_fa)
    imgProcess = img_nl_fa





def near():
    global value_nearest
    value_nearest=StringVar()
    c_nearest=Toplevel()
    Label(c_nearest,text='请选择缩放倍数 :',width=20).grid(row=0,column=0)
    sca=Scale(c_nearest,from_=-4,to=4,orient=HORIZONTAL,variable=value_nearest).grid(row=1,column=0)
    Button(c_nearest,text='确定',command=nearest).grid(row=2,column=0)

def nearest():
    global imgProcess
    num =  int(value_nearest.get())
    multiple=[0.1,0.25,0.5,0.75,1,1.5,2,2.5,3]
    multi=multiple[num+4]
    new_width = int(width * multi)
    new_height = int(height * multi)
    nearest_img=Image.new('RGB',(new_width,new_height))
    L_img=img
    pix_nearest=nearest_img.load()
    pix_img=L_img.load()
    for i in range(new_width):
        for j in range(new_height):
            x  = int(i/multi)
            y = int(j/multi)
            pix_nearest[i,j]=pix_img[x,y]
    show_pic_process(nearest_img)
    show_hist_process(nearest_img)
    imgProcess = nearest_img

    
def bili():
    global value_bili
    value_bili=StringVar()
    c_nearest=Toplevel()
    Label(c_nearest,text='请选择缩放倍数 :',width=20).grid(row=0,column=0)
    sca=Scale(c_nearest,from_=-4,to=4,orient=HORIZONTAL,variable=value_bili).grid(row=1,column=0)
    Button(c_nearest,text='确定',command=bilinear).grid(row=2,column=0)

def bilinear():
    global imgProcess
    num= int(value_bili.get())
    multiple=[0.1,0.25,0.5,0.75,1,1.5,2,2.5,3]
    multi=multiple[num+4]
    new_width = int(width*multi)
    new_height = int(height*multi)
    bili_img=Image.new('RGB',(new_width,new_height))
    pix_bili=bili_img.load()
    pix_img=img.load()
    for i in range(new_width):
        for j in range(new_height):
            x=float(i)/multi
            y=float(j)/multi
            u= x - int(x)
            v= y - int(y)
            if int(x) ==width-1 or int(y)== height-1:
                pix_bili[i,j]=pix_img[int(x),int(y)]
            else:
                pix_r=(1-u)*(1-v)*pix_img[int(x),int(y)][0]+(1-u)*v*pix_img[int(x),int(y)+1][0]+u*(1-v)*pix_img[int(x)+1,int(y)][0]+u*v*pix_img[int(x)+1,int(y)+1][0]
                pix_g=(1-u)*(1-v)*pix_img[int(x),int(y)][1]+(1-u)*v*pix_img[int(x),int(y)+1][1]+u*(1-v)*pix_img[int(x)+1,int(y)][1]+u*v*pix_img[int(x)+1,int(y)+1][1]
                pix_b=(1-u)*(1-v)*pix_img[int(x),int(y)][2]+(1-u)*v*pix_img[int(x),int(y)+1][2]+u*(1-v)*pix_img[int(x)+1,int(y)][2]+u*v*pix_img[int(x)+1,int(y)+1][2]
                pix_bili[i,j]=(int(pix_r),int(pix_g),int(pix_b))
    show_pic_process(bili_img)
    show_hist_process(bili_img)
    imgProcess = bili_img


def trans():
    global value_trans
    value_trans=StringVar()
    c_translation=Toplevel()
    Label(c_translation,text='请选择平移百分比 :',width=20).grid(row=0,column=0)
    sca=Scale(c_translation,from_=-100,to=100,orient=HORIZONTAL,variable=value_trans).grid(row=1,column=0)
    Button(c_translation,text='确定',command=translate).grid(row=2,column=0)

def translate():
    global imgProcess
    translating=int(value_trans.get())
    img_trans=Image.new('RGB',(width,height))
    pix_pre=img.load()
    pix_trans=img_trans.load() 
    for i in range(width):
        for j in range(height):
            i_trans=int((i-width*translating/100))%width
            pix_trans[i,j]=pix_pre[i_trans,j]
    show_pic_process(img_trans)
    show_hist_process(img_trans)
    imgProcess = img_trans




def rot():
    global value_rot
    value_rot=StringVar()
    c_rotating=Toplevel()
    Label(c_rotating,text='choose: ',width=20).grid(row=0,column=0)
    sca=Scale(c_rotating,from_=-360,to=360,orient=HORIZONTAL,variable=value_rot).grid(row=1,column=0)
    Button(c_rotating,text='ok',command=rotating).grid(row=2,column=0)


def rotating():
    global imgProcess
    angle=-int(value_rot.get())
    img_rotating=img.rotate(angle)
    show_pic_process(img_rotating)
    show_hist_process(img_rotating)
    imgProcess = img_rotating







def sampling():
    '''采样和量化函数


    可叠加处理采样和量化，显示处理后图像'''
    global imgProcess
    if filepath_pre=='':
        showwarning(title='Error!',
                    message='Please Open a Picture!'
                    )
    else:
        global img_sampling
        img_sampling=Image.new('L',img.size)
        img_pre=img.convert('L')
        pix_pre=img_pre.load()
        pix_sampling=img_sampling.load()
        if vSimple.get()!='':
           sampling=int(vSimple.get())
        else:
            sampling=1
        if vQuantify.get()!='':
            quantity=int(vQuantify.get())
        else:
            quantity=256
        for i in range(width):
            for j in range(height):
                pix_sampling[i,j]=pix_pre[i-i%sampling,j-j%sampling]
        for i in range(width):
            for j in range(height):
                pix_sampling[i,j]=int(pix_sampling[i,j]*quantity/256)*256/(quantity-1)
        show_pic_process(img_sampling)
        show_hist_process(img_sampling)
        imgProcess = img_sampling
    

#****************************************
#真彩色转256色图像
#包括 palette,pair,bmpconvert
def palette():
    global rgb_large
    global bit_tuple
    pre_img = img.load()
    rgb_large=[]
    rgb_tuple = []
    rgb_num = []
    bit_tuple = []
    for i in range(width):
        for j in range(height):    
            r_large = bin(pre_img[i,j][0])[:2]+'00000000'+bin(pre_img[i,j][0])[2:]
            g_large = bin(pre_img[i,j][1])[:2]+'00000000'+bin(pre_img[i,j][1])[2:]
            b_large = bin(pre_img[i,j][2])[:2]+'00000000'+bin(pre_img[i,j][2])[2:]
            rgb = r_large[-8:-4] + g_large[-8:-4] + b_large[-8:-4]
            rgb_large.append(rgb)
    rgb_tuple = list(set(rgb_large))
    for item in rgb_tuple:
        rgb_num.append(rgb_large.count(item))
    for k in range(256):    
        numnum = max(rgb_num)
        rgb_max = rgb_tuple[rgb_num.index(max(rgb_num))] 
        bit_tuple.append(rgb_max)
        rgb_tuple.remove(rgb_max)
        rgb_num.remove(numnum)
    pair()
    bmpconvert()

def pair():
    global dic_pair
    pix_img = img.load()
    dic_pair = {}
    rgb_new = list(set(rgb_large))
    for rgb_item in rgb_new:
        pair_min =99999999999999
        r_item = int(rgb_item[:4]+'0000',2)
        g_item = int(rgb_item[4:8]+'0000',2)
        b_item = int(rgb_item[8:]+'0000',2)
        for bit_item in bit_tuple:
            r_bit = int(bit_item[:4]+'0000',2)
            g_bit = int(bit_item[4:8]+'0000',2)
            b_bit = int(bit_item[8:]+'0000',2)
            pair_num = (r_item-r_bit)**2 + (g_item-g_bit)**2 + (b_item-b_bit)**2
            if pair_num <= pair_min:
                pair_min = pair_num
                pair_bit = bit_item
        dic_pair.setdefault(rgb_item,pair_bit)

def bmpconvert():
    bmp_img = Image.new('RGB',img.size)
    pix_bmp = bmp_img.load()
    pix_img = img.load()
    for i in range(width):
        for j in range(height):
            #rgb_img = rgb_large[width*i+j]
            #rgb_bmp = dic_pair[rgb_img]
            r_img = bin(pix_img[i,j][0])[:2]+'00000000'+bin(pix_img[i,j][0])[2:]
            g_img = bin(pix_img[i,j][1])[:2]+'00000000'+bin(pix_img[i,j][1])[2:]
            b_img = bin(pix_img[i,j][2])[:2]+'00000000'+bin(pix_img[i,j][2])[2:]
            rgb_img = r_img[-8:-4] + g_img[-8:-4] + b_img[-8:-4]
            rgb_bmp = dic_pair[rgb_img]
            r_bmp = int(rgb_bmp[:4]+'0000',2)
            g_bmp = int(rgb_bmp[4:8]+'0000',2)
            b_bmp = int(rgb_bmp[8:]+'0000',2)
            pix_bmp[i,j] = (r_bmp, g_bmp, b_bmp)
    show_pic_process(bmp_img)
    show_hist_process(bmp_img)


            
    
    
    
def bitplane():
    '''位平面图函数

    显示图像的8个位平面图'''
    if filepath_pre=='':
        showwarning(title='Error!',
                    message='Please open a picture!'
                    )
    else:
        global img_bitplane
        img_bitplane=Image.new('L',img.size)
        img_pre=img.convert('L')
        img_new=Image.new('L',(600,600))
        pix_pre=img_pre.load()
        pix_bitplane=img_bitplane.load()
        for plane in range(8):
            for i in range(width):
                for j in range(height):
                    pix_len=len(bin(pix_pre[i,j]))
                    if (plane>pix_len-3):
                        pix_now=0
                    else:
                        pix_now=int(bin(pix_pre[i,j])[pix_len-plane-1])
                    if (pix_now):
                        pix_now=255
                    pix_bitplane[i,j]=pix_now
            if (plane==0):
                box=(0,1,148,200)
            if (plane==1):
                box=(150,1,298,200)
            if (plane==2):
                box=(300,1,448,200)
            if (plane==3):
                box=(450,1,598,200)
            if (plane==4):
                box=(0,201,148,400)
            if (plane==5):
                box=(150,201,298,400)
            if (plane==6):
                box=(300,201,448,400)
            if (plane==7):
                box=(450,201,598,400)
            region=img_bitplane.resize((148,199))
            img_new.paste(region,box)
        pic_bitplane=Toplevel()
        cv_bitplane=Canvas(pic_bitplane,width=600,height=401)
        img_process=ImageTk.PhotoImage(img_new)
        cv_bitplane.create_image((300,300),image=img_process)
        cv_bitplane.img_process=img_process
        cv_bitplane.pack()


def hist_process(img):
    '''直方图函数


    生成图像的直方图，并计算直方图具体信息'''
    global grey_avg
    global grey_mid
    global pix_all
    global pix_dev                           
    pix_dev=0
    hist=[0 for i in range(256)]
    width,height=img.size
    pix_all=width*height
    img_hist=Image.new('L',(256*2,100))
    hist_pix=img_hist.load()
    img_L=img.convert('L')
    img_pix=img_L.load()
    pix_sum=0
    for i in range(width):
        for j in range(height):
            hist[img_pix[i,j]]+=1
            pix_sum+=img_pix[i,j]
    max_hist=max(hist)
    for i in range(256):
        pix_height=(hist[i]*100)/max_hist
        for j in range(100-pix_height):
            hist_pix[i*2,j]=255
            hist_pix[i*2+1,j]=255
    pix_place=pix_all/2
    pix_num=0
    for i in range(256):
        pix_num+=hist[i]
        if(pix_num>pix_place):
            grey_mid=i
            break
    grey_avg=pix_sum/pix_all
    for i in range(width):
        for j in range(height):
            pix_dev+=(img_pix[i,j]-grey_avg)**2
    pix_dev=(float(pix_dev)/pix_all)**0.5
    return img_hist
    



def info():
    '''图像信息函数


    读取jpeg图像文件头，计算并显示文件头信息'''
    if filepath_pre=='':
        showwarning(title='Error!',
                    message='Please Open a Picture!'
                    )
    else:
        info_pic=open(filepath_pre,'rb')
        info_pic.read(11)
        version_pic=str(ord(info_pic.read(1)))+'.'+str(ord(info_pic.read(1)))
        while(1):
            if(info_pic.read(1)=='\xFF'):
                if(info_pic.read(1)=='\xc0'):
                    info_pic.read(2)
                    jdu_pic=str(ord(info_pic.read(1)))
                    height_pic=str(ord(info_pic.read(1))*16*16+ord(info_pic.read(1)))
                    width_pic=str(ord(info_pic.read(1))*16*16+ord(info_pic.read(1)))
                    num_pic=str(ord(info_pic.read(1)))
                    break
        info_pic.close()
        information=Toplevel()
        infor=Label(information,text=('版本号: '+version_pic+'\n'
                                      +'图像精度: '+jdu_pic+'\n'
                                      +'图像高度: '+height_pic+'\n'
                                      +'图像宽度: '+width_pic+'\n'
                                      +'颜色分量数: '+num_pic))
        infor.pack()


#---------
root=Tk()
root.title('数字图像处理')
menubar=Menu(root)


#图片选项
Picmenu=Menu(menubar,tearoff=0)
Picmenu.add_command(label='打开',command=choose_pic)
Picmenu.add_command(label='另存为',command=save_pic) 
Picmenu.add_command(label='保存',command=saved)
menubar.add_cascade(label='图片',menu=Picmenu)




menubar.add_command(label='灰度化',command=convert_L)

#点运算选项
vLinear_enhance=StringVar()
vLinear_fade=StringVar()
vnLinear_enhance=StringVar()
vnLinear_fade=StringVar()
Oper_menu=Menu(menubar,tearoff=0)
Linear_enhance=Menu(Oper_menu,tearoff=0)
Linear_fade=Menu(Oper_menu,tearoff=0)
nLinear_enhance=Menu(Oper_menu,tearoff=0)
nLinear_fade=Menu(Oper_menu,tearoff=0)
for i in ['1.1','1.2','1.3','1.4','1.5']:
    Linear_enhance.add_radiobutton(label=i,variable=vLinear_enhance,command=linear_en)
    nLinear_enhance.add_radiobutton(label=i,variable=vnLinear_enhance,command=nlinear_en)
for j in ['0.9','0.8','0.7','0.6','0.5']:
    Linear_fade.add_radiobutton(label=j,variable=vLinear_fade,command=linear_fa)
    nLinear_fade.add_radiobutton(label=j,variable=vnLinear_fade,command=nlinear_fa)
Oper_menu.add_command(label='均衡化',command=hist_equ)
Oper_menu.add_cascade(label='线性增强',menu=Linear_enhance)
Oper_menu.add_cascade(label='线性减弱',menu=Linear_fade)
Oper_menu.add_cascade(label='非线性增强',menu=nLinear_enhance)
Oper_menu.add_cascade(label='非线性减弱',menu=nLinear_fade)
menubar.add_cascade(label='点运算',menu=Oper_menu)





#几何运算选项
Geomenu=Menu(menubar,tearoff=0)
zoom=Menu(Geomenu,tearoff=0)
rotate=Menu(Geomenu,tearoff=0)
translation=Menu(Geomenu,tearoff=0)
Geomenu.add_command(label='旋转',command=rot)
Geomenu.add_command(label='平移',command=trans)
zoom.add_radiobutton(label='最邻近插值',command=near)
zoom.add_radiobutton(label='双线性插值',command=bili)
Geomenu.add_cascade(label='缩放',menu=zoom)
menubar.add_cascade(label='几何运算',menu=Geomenu)

#图像增强
H1 = [1.0/9]*9
H2 = [1.0/10]*9
H2[4] = 0.2
H3 = [1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16]
H4 = [1.0/8]*9
H4[4] = 0
H5 = [0, -1, 0, -1, 5, -1, 0, -1, 0]
H6 = [-1, -1, -1, -1, 9, -1, -1, -1, -1]
form_smooth = [ H1, H2, H3, H4]
form_sharp = [H5, H6]
vSmooth=StringVar()
vSharp = StringVar()
enmenu=Menu(menubar,tearoff=0)
smooth=Menu(enmenu,tearoff=0)
sharp = Menu( enmenu, tearoff = 0)
form_name=['H1','H2','H3','H4']
for i in range(4):
    smooth.add_radiobutton(label=form_name[i],variable=vSmooth,command=smoothen)
for j in range(2):
    sharp.add_radiobutton(label=form_name[j],variable=vSharp,command=sharpen)
enmenu.add_cascade(label = '图像平滑', menu= smooth)
enmenu.add_cascade(label = '图像锐化', menu = sharp)
enmenu.add_command(label = '图像卷积', command = c_convolute)
menubar.add_cascade(label = '图像增强', menu=enmenu)

#图像分割
cutmenu = Menu(menubar, tearoff = 0)
cutmenu.add_command(label = '拉普拉斯算子',command = edge_laplace)
cutmenu.add_command(label = 'Kirsch方向算子',command = edge_kirsch)
menubar.add_cascade(label = '图像分割', menu = cutmenu)




#转256色
bitmenu = Menu(menubar, tearoff = 0)
#bitmenu.add_command(label = '调色版',command = palette)
bitmenu.add_command(label = '真彩色转256色',command = palette)
menubar.add_cascade(label = '彩色图像处理', menu = bitmenu)


#霍夫变换
houghmenu = Menu(menubar, tearoff = 0)
houghmenu.add_command(label = '图像细化', command = thin)
houghmenu.add_command(label = '霍夫变换检测直线', command = lineChoose)
houghmenu.add_command(label = '检测圆形', command = circleCheck)
menubar.add_cascade(label = '数学形态学', menu = houghmenu)

#yasuo
compressMenu = Menu(menubar, tearoff = 0)
compressMenu.add_command(label = 'Huffman', command = huffman)
compressMenu.add_command(label = 'Fano', command = fano)
compressMenu.add_command(label = 'Rle')
compressMenu.add_command(label = 'Figure')
menubar.add_cascade(label = 'yasuo', menu = compressMenu)


#Opencv进行图像处理
cvmenu=Menu(menubar,tearoff=0)
cvmenu.add_command(label='傅立叶变换',command= four)
cvmenu.add_command(label='离散余弦变换',command=lisanyuxian)
cvmenu.add_command(label='Canny边缘检测',command = canny)
menubar.add_cascade(label='Opencv',menu=cvmenu)



#采样和量化选项
vSimple=StringVar()
vQuantify=StringVar()
Calmenu=Menu(menubar,tearoff=0)
Simple=Menu(Calmenu,tearoff=0)
Quantify=Menu(Calmenu,tearoff=0)
for i in ['1','2','4','8','16']:
    Simple.add_radiobutton(label=i,variable=vSimple,command=sampling)
for j in ['256','128','64','32','16','8','4','2']:
    Quantify.add_radiobutton(label=j,variable=vQuantify,command=sampling)
Calmenu.add_cascade(label='采样',menu=Simple)
Calmenu.add_cascade(label='量化',menu=Quantify)
menubar.add_cascade(label='采样和量化',menu=Calmenu)

menubar.add_command(label='位平面图',command=bitplane)
menubar.add_command(label='图片信息',command=info)

root['menu']=menubar


cv_pri=Canvas(root, width=512,height=512)
cv_process=Canvas(root,width=512,height=512)
cv_pri.grid(row=0,column=0)
cv_process.grid(row=0,column=1)

cv_hist_pri=Canvas(root,width=512,height=100)
cv_hist_process=Canvas(root,width=512,height=100)
cv_hist_pri.grid(row=1,column=0)
cv_hist_process.grid(row=1,column=1)

Lab_info_pri=Label(root,width=64,height=5)
Lab_info_process=Label(root,width=64,height=5)
Lab_info_pri.grid(row=2,column=0)
Lab_info_process.grid(row=2,column=1)




root.mainloop()


