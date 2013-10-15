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
    if filepath!= '':
        filepath_pre=filepath
        img=Image.open(filepath)
        width,height=img.size
        show_pic(img)
        show_pic_process(img)

#显示原图
def show_pic(img):
    im=ImageTk.PhotoImage(img)
    cv_pri.create_image((300,256),image=im)
    cv_pri.im=im
    
#显示处理后图像
def show_pic_process(img):
    global im_process
    im_process=img
    img_process=ImageTk.PhotoImage(im_process)
    cv_process.create_image((300,256),image=img_process)
    cv_process.img_process=img_process

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

menubar.add_command(label='图片信息',command=info)

root['menu']=menubar


cv_pri=Canvas(root, width=600,height=512,bg='grey')
cv_process=Canvas(root,width=600,height=512,bg='grey')
cv_pri.grid(row=0,column=0)
cv_process.grid(row=0,column=1)

root.mainloop()


