# -*- coding: utf-8 -*-
from Tkinter import*
from PIL import Image,ImageTk
import tkFileDialog
from tkMessageBox import*
import os
import sys



filepath=''

#选择图像
def choose_pic():
    global filepath
    global img
    global im
    filepath=tkFileDialog.askopenfilename(initialdir = 'C:\Users\Public\Pictures')
    if filepath!= '':
        img=Image.open(filepath)
        show_pic(img)

#显示图像
def show_pic(img):
    im=ImageTk.PhotoImage(img)
    cv.create_image((400,300),image=im)
    cv.im=im

#保存图像
def save_pic():
    global img
    save_filepath=tkFileDialog.asksaveasfilename(defaultextension='.jpg',filetypes=([('JPG','*.jpg'),('PNG','*.png'),('GIF','*.gif')]))
    #print(save_filepath)
    img.save(save_filepath)

    
#图像信息
def info():
    if filepath=='':
        showwarning(title='Error!',
                    message='Please Open a Picture!'
                    )
    else:
        info_pic=open(filepath,'rb')
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



root=Tk()
menubar=Menu(root)
Picmenu=Menu(menubar,tearoff=0)
Picmenu.add_command(label='打开',command=choose_pic)
Picmenu.add_command(label='保存',command=save_pic) 
menubar.add_cascade(label='图片',menu=Picmenu)

menubar.add_command(label='图片信息',command=info)
root['menu']=menubar

cv=Canvas(root, width=800,height=600)
cv.pack()

root.mainloop()


