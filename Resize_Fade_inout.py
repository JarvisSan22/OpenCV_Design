# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:16:38 2021

@author: Jarvi
"""

#from pyxelate_orignial.pyxelate import Pyxelate
#from pyxelate_orignial.pyx  import Pyx
from skimage import io
#import matplotlib.pyplot as plt
import cv2
import glob 
import pandas as pd 
import os
from PIL import Image
import numpy as np

"""
def Pixlize(img,name,folder,p=None,factor = 6,colors = 8,dither =True,plot=True):
    
    savename=os.path.join(folder,name+"_factor_{}_color_{}_pixle.jpg".format(factor,colors))
    print(savename)
    #if os.path.exist(savename):
     #   img_small=io.imread(savename)
 #   else:
    height, width, _ = img.shape 
    if p:
        pass
    else:
        p = Pyxelate(height // factor, width // factor, colors, dither)
    img_small = p.convert(img)  # convert an image with these settings
    if plot:
        _, axes = plt.subplots(1, 2, figsize=(16, 16))
        axes[0].imshow(img)
        axes[1].imshow(img_small)
        plt.show()
        for ax in axes:
            ax.axis("off")

    cv2.imwrite(savename,img_small)
   
    return img_small,savename,p
"""
def ResizePixilate(img,factor):
    
    h,w,c=img.shape
    img_small=cv2.resize(img,(w//factor,h//factor),200)
    
    img_pix=cv2.resize(img_small,(w,h),interpolation = cv2.INTER_AREA)
    
    return img_pix

    


def PixleFadeInOut(img1,img2,loops=2,frames_n=50,FadeType="Log",FadeFramesList=[1,20,40],savefolder="pic/",outname="test.gif",cleanframes=True):
    h,w,c=img1.shape
    img2=cv2.resize(img2,(w,h),interpolation = cv2.INTER_AREA)
    max_resize=100
    frames_n=frames_n
    InOutFade=FadeFramesList #[Fade in start, fade in stop fadeout start, fade out stop ]

    #fade in out factors list 
    fac_list=np.zeros(frames_n)
    
    #Linear 
    if FadeType=="Linear":
        fac_list[InOutFade[0]:InOutFade[1]]=np.linspace(1,max_resize,InOutFade[1]-InOutFade[0])
        fac_list[InOutFade[1]:InOutFade[2]]=np.linspace(max_resize,1,InOutFade[2]-InOutFade[1])
    elif FadeType=="Log":
        fac_list[InOutFade[0]:InOutFade[1]]=10*np.logspace(0,1,InOutFade[1]-InOutFade[0],base=10,endpoint=False)
    
        fac_list[InOutFade[1]:InOutFade[2]]=10*np.logspace(1,0,InOutFade[2]-InOutFade[1],base=10,endpoint=False)
        
    
    
    fac_list[InOutFade[2]:]=1
    #gif loops 

    #dummy vars 
    frames=[]
    frameid=[]
    #weightings 
    fadeoutWeight=0
    fadeoutRate=0.1
    #check folder exists 
    for n,fac in zip(range(0,frames_n),fac_list):
        fac=int(fac)
        print("fac:",fac)
        savename=savefolder+"/"+"frame_{}.jpg".format(n)
        
        if n>=InOutFade[2]:
            saveimg=img2
            print(n,frames_n,"img2")
        elif n>=InOutFade[1]:
            saveimg=img1
            
            if fadeoutWeight>=1:
                saveimg= img2
                print(n,frames_n,"Fade out img2 ")
            else:
                saveimg=cv2.addWeighted( img1, 1-fadeoutWeight, img2, fadeoutWeight, 0)
                fadeoutWeight+=fadeoutRate
                print(n,frames_n,"Fade out Img 2")
            
            saveimg=ResizePixilate(saveimg,fac)
            
        elif n>=InOutFade[0]:
            if fac==0:
                fac=1
            
            saveimg=img1
            saveimg=ResizePixilate(saveimg,fac)
            
            print(n,frames_n,"Fade in img1")   
        else:
            saveimg=img1
            print(n,frames_n,"img1")
        
            
        #save
        cv2.imwrite(savename,cv2.cvtColor(saveimg,cv2.COLOR_RGB2BGR))
        frames.append(savename)
        frameid.append(n)
    
    frame_rate=10

    name=os.path.join(savefolder,outname)
     
    
    #Create gif
    if ".gif" not in name:
        name+=".gif" 
    Images=pd.DataFrame({"ID":frameid,"images":frames})
    Images=Images.sort_values("ID")
    if loops>0:
        Images_new=Images.copy()
        for l in range(1,loops):
            if l % 2 ==0:
                Images_new=pd.concat([Images_new,Images],axis=0,ignore_index=True)
            else:
                Images_new=pd.concat([Images_new,Images[::-1]],axis=0,ignore_index=True)
        Images_new["ID"]=Images_new.index
        Images=Images_new
                
   
    
    frames_n=len(Images)
    images = list(map(lambda file: Image.open(file), Images["images"]))
    images[0].save(name, save_all=True, append_images=images[1:], duration=2*frames_n, loop=loops) 

    #Del Used frames 
    if cleanframes:
        for frame in frames:
            os.remove(frame)
    return Images


def main():
    file1=os.path.join( os.getcwd(), 'examples', 'sakura_16.jpg' )
    img1=io.imread(file1)
    #img_pix,savename,p=Pixlize(img,"sakura_16",folder,p=None,factor = 2,colors = 8,dither =True,plot=True)
    file2=os.path.join( os.getcwd(), 'examples', 'sakura_16_factor_2_color_8_pixle.jpg' )
    img2=cv2.cvtColor(io.imread(file2),cv2.COLOR_RGB2BGR)
    Images=PixleFadeInOut(img1,img2,loops=2,frames_n=50,FadeFramesList=[5,20,30],savefolder="examples/",outname="test.gif",cleanframes=False)
    return Images

if __name__=='__main__':
    Images=main()
    #Images=pd.concat([Images,Images[::-1]],ignore_index=True,axis=0)
    #frames_n=len
    #name="test2.gif" 
    #images[0].save(name, save_all=True, append_images=images[1:], duration=2*frames_n, loop=loops) 
