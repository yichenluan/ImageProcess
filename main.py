#-*- coding: utf-8 -*-
from Tkinter import*
from PIL import Image,ImageTk
import tkFileDialog
from tkMessageBox import*
import cv
import cv2
import numpy as np

filepath=''
filepath_pre=''



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
    '''保存图像函数

    在弹出框中选择保存路径和格式'''
    save_filepath=tkFileDialog.asksaveasfilename(defaultextension='.jpg',filetypes=([('JPG','*.jpg'),('PNG','*.png'),('GIF','*.gif')]))
    im_process.save(save_filepath)


##
#图像增强
#
#包括: convolution, smoothen, sharpen, c_convolute, convolute  

def convolution():
    '''计算卷积函数


    根据给定的模板数组，得到卷积后的图像'''
    global img_con
    pix_pre = img.load()
    img_con = Image.new('RGB',img.size)
    pix_con = img_con.load()
    for i in range(width):
        for j in range(height):
            if i in [0,width-1] or j in [0,height-1]:
                 pix_con[i,j] = pix_pre[i,j]
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
                pix_con[i,j] = (int(sum_r), int(sum_g), int(sum_b))


def smoothen():
    '''图像平滑


    从Menu中得到模板数组，传递给卷积函数，显示平滑后图像'''
    global mod
    model = int(vSmooth.get()[-1])
    mod = form_smooth[model-1] 
    convolution()
    show_pic_process(img_con)
    show_hist_process(img_con)


def sharpen():
    '''图像锐化

    从Menu中得到模板数组，传递给卷积函数，显示锐化后图像'''
    global mod
    model = int(vSharp.get()[-1])
    mod = form_sharp[model-1]
    convolution()
    show_pic_process(img_con)
    show_pic_process(img_con)

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
    global mod
    array = value_matrix.get().split()
    mod = [int(num) for num in array]
    convolution()
    show_pic_process(img_con)
    show_pic_process(img_con)
     


def convert_L():
    '''灰度化函数
    根据原图的R，G，B分量得到灰度值
    显示灰度后图像'''
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
    #cv.ShowImage('iAfter',iAfter)
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




def hist_equ():
    '''均衡化函数


    将灰度图均衡化后显示'''
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
                




#图像线性增强
def linear_en():
    multiple=float(vLinear_enhance.get())
    img_l_en=img.point(lambda i: i*multiple)
    show_pic_process(img_l_en)
    show_hist_process(img_l_en)


#图像线性减弱
def linear_fa():
    multiple=float(vLinear_fade.get())
    img_l_fa=img.point(lambda i:i*multiple)
    show_pic_process(img_l_fa)
    show_hist_process(img_l_fa)

#图像非线性增强
def nlinear_en():
    multiple=float(vnLinear_enhance.get())
    img_nl_en=img.point(lambda i:(i+i*multiple*(255-i)/255))
    show_pic_process(img_nl_en)
    show_hist_process(img_nl_en)

#图像非线性减弱
def nlinear_fa():
    multiple=float(vnLinear_fade.get())
    img_nl_fa=img.point(lambda i:(i+i*multiple*(255-i)/255))
    show_pic_process(img_nl_fa)
    show_hist_process(img_nl_fa)



#最邻近插值

def near():
    global value_nearest
    value_nearest=StringVar()
    c_nearest=Toplevel()
    Label(c_nearest,text='请选择缩放倍数 :',width=20).grid(row=0,column=0)
    sca=Scale(c_nearest,from_=-4,to=4,orient=HORIZONTAL,variable=value_nearest).grid(row=1,column=0)
    Button(c_nearest,text='确定',command=nearest).grid(row=2,column=0)

def nearest():
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

#双线性插值    
def bili():
    global value_bili
    value_bili=StringVar()
    c_nearest=Toplevel()
    Label(c_nearest,text='请选择缩放倍数 :',width=20).grid(row=0,column=0)
    sca=Scale(c_nearest,from_=-4,to=4,orient=HORIZONTAL,variable=value_bili).grid(row=1,column=0)
    Button(c_nearest,text='确定',command=bilinear).grid(row=2,column=0)

def bilinear():
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

#平移
def trans():
    global value_trans
    value_trans=StringVar()
    c_translation=Toplevel()
    Label(c_translation,text='请选择平移百分比 :',width=20).grid(row=0,column=0)
    sca=Scale(c_translation,from_=-100,to=100,orient=HORIZONTAL,variable=value_trans).grid(row=1,column=0)
    Button(c_translation,text='确定',command=translate).grid(row=2,column=0)

def translate():
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



#旋转，目前直接调用库函数
def rot():
    global value_rot
    value_rot=StringVar()
    c_rotating=Toplevel()
    Label(c_rotating,text='choose: ',width=20).grid(row=0,column=0)
    sca=Scale(c_rotating,from_=-360,to=360,orient=HORIZONTAL,variable=value_rot).grid(row=1,column=0)
    Button(c_rotating,text='ok',command=rotating).grid(row=2,column=0)

def rotating():
    angle=-int(value_rot.get())
    img_rotating=img.rotate(angle)
    show_pic_process(img_rotating)
    show_hist_process(img_rotating)







def sampling():
    '''采样和量化函数


    可叠加处理采样和量化，显示处理后图像'''
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
Picmenu.add_command(label='保存',command=save_pic) 
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




#Opencv进行图像处理
cvmenu=Menu(menubar,tearoff=0)
cvmenu.add_command(label='傅立叶变换',command= four)
cvmenu.add_command(label='离散余弦变换',command=lisanyuxian)
menubar.add_cascade(label='图像变换',menu=cvmenu)



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




if __name__ == '__main__':
    root.mainloop()



