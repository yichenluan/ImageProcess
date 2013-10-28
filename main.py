#-*- coding: utf-8 -*-
from Tkinter import*
from PIL import Image,ImageTk
import tkFileDialog
from tkMessageBox import*


filepath=''
filepath_pre=''

#选择图像
def choose_pic():
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

#显示原图
def show_pic(img):
    im=ImageTk.PhotoImage(img)
    cv_pri.create_image((256,256),image=im)
    cv_pri.im=im
    
#显示处理后图像
def show_pic_process(img):
    global im_process
    im_process=img
    img_process=ImageTk.PhotoImage(im_process)
    cv_process.create_image((256,256),image=img_process)
    cv_process.img_process=img_process


#显示原图灰度直方图
def show_hist(im):
    img_show=hist_process(im)
    img=ImageTk.PhotoImage(img_show)
    cv_hist_pri.create_image((256,50),image=img)
    cv_hist_pri.img=img
    st='总像素: '+str(pix_all)+'  '+'平均灰度: '+str(grey_avg)+'   '+'中值灰度: '+str(grey_mid)+'  '+'标准差: '+str(pix_dev)[0:5]
    Lab_info_pri.configure(text=st)
    Lab_info_pri.text=st
    

#显示处理后图像灰度直方图
def show_hist_process(im):
    img_show=hist_process(im)
    img=ImageTk.PhotoImage(img_show)
    cv_hist_process.create_image((256,50),image=img)
    cv_hist_process.img=img
    st='总像素: '+str(pix_all)+'    '+'平均灰度: '+str(grey_avg)+'  '+'中值灰度: '+str(grey_mid)+'  '+'标准差: '+str(pix_dev)[0:5]
    Lab_info_process.configure(text=st)
    Lab_info_process.st=st



#保存图像
def save_pic():
    save_filepath=tkFileDialog.asksaveasfilename(defaultextension='.jpg',filetypes=([('JPG','*.jpg'),('PNG','*.png'),('GIF','*.gif')]))
    im_process.save(save_filepath)

#灰度化
def convert_L():
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


#均衡化
def hist_equ():
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


#采样和量化
def sampling():
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
    
#位平面图
def bitplane():
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


#生成直方图
def hist_process(img):
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
    



#图像信息
def info():
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
Picmenu=Menu(menubar,tearoff=0)
Picmenu.add_command(label='打开',command=choose_pic)
Picmenu.add_command(label='保存',command=save_pic) 
menubar.add_cascade(label='图片',menu=Picmenu)

menubar.add_command(label='灰度化',command=convert_L)

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

Geomenu=Menu(menubar,tearoff=0)
menubar.add_cascade(label='几何运算',menu=Geomenu)


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


