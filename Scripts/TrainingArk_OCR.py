# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 23:05:19 2022

@author: Kenneth
"""
#%%
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import os
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY']= '0'
tf.autograph.experimental.do_not_convert

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preProcess(img):
    
    gray=get_grayscale(img)
    thresh=thresholding(gray)
    inverted=cv2.bitwise_not(thresh)

    return inverted

def remove_noise(image):
    return cv2.medianBlur(image,5)


def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)


def strip_bars(img):
    
#    denoise=remove_noise(img)
    
    gray_base = get_grayscale(img)
#    thresh=thresholding(gray_base)#use for removing, need to threshold because not always percet cut between black and color

    if np.sum(gray_base[0:100,500:600])==0:#check if the top bit is a blackbar

        col=gray_base[:,100]
        
        non_black=col!=0
        
        changepts=[]
        
        for count,each in enumerate(non_black):
            if count==0:
                prev=False
                continue
            else: 
                if each!=prev:
                    changepts.append(count)
                    prev=each
                else:
                    prev=each
                    
        cropped_base=img[changepts[0]:changepts[-1],:,:]
        
        return cropped_base
    else:
        return img
    
#returns true if it detects "mmr" or "ranked" false otherwise
# will probably be folded into general scoreboard detection or win/loss detection, but seperate for now
def detect_ranked_result(img):
    custom_config = r'--psm 3'
#    inverted=preProcess(img)
    
    wl_location=[0.35,0.05]

    x=round(wl_location[0]*base_res[0])
    x1=round(wl_location[0]*base_res[0])+1000
    y=round(wl_location[1]*base_res[1])
    y1=round(wl_location[1]*base_res[1])+150
    
    sub_image=img[y:y1,x:x1]
    
    
    d=pytesseract.image_to_data(sub_image, output_type=Output.DICT, config=custom_config)
    
    
    ranked=False
    match_result='None'
    
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if str.lower(d['text'][i])=='ranked' or str.lower(d['text'][i])=='mmr':
            ranked=True
        
        if str.lower(d['text'][i])=='win':
            match_result='Win'
        
        if str.lower(d['text'][i])=='lose':
            match_result='Lose'
        
        
    return ranked, match_result

#def make_model(input_shape,num_classes): 1.29 model
#    
#    data_augmentation=keras.Sequential([
#        
#        layers.RandomFlip("horizontal"),
#        layers.RandomRotation(0.1),
#        ]
#    )
#    
#    inputs = keras.Input(shape=input_shape)
#    x = data_augmentation(inputs)
#    
#    x=layers.Conv2D(4, kernel_size=(25, 25))(x)
#    x=layers.MaxPooling2D((5,5),strides=3)(x)
#    x=layers.BatchNormalization()(x)
#    x=layers.Activation("relu")(x)
#    
#    x=layers.Conv2D(8, kernel_size=(5, 5))(x)
#    x=layers.BatchNormalization()(x)
#    x=layers.Activation("relu")(x)
#    
#    x=layers.Conv2D(16, kernel_size=(2, 2))(x)
#    x=layers.BatchNormalization()(x)
#    x=layers.Activation("relu")(x)
#    
#    x=layers.Flatten()(x)
#    
#    x=layers.Dense(48,activation="relu")(x)
#    x=layers.BatchNormalization()(x)
#    
#    x=layers.Dropout(0.5)(x)
#    outputs=layers.Dense(num_classes,activation="softmax")(x)
#    
#    return keras.Model(inputs,outputs)


def make_model(input_shape,num_classes): #2.3 model
    
    data_augmentation=keras.Sequential([
    
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    ]
    )
    
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    x=layers.Conv2D(4, kernel_size=(25, 25))(x)
    x=layers.MaxPooling2D((5,5),strides=3)(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    
    x=layers.Conv2D(16, kernel_size=(5, 5))(x)
    x=layers.MaxPooling2D((3,3))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    
    x=layers.Conv2D(64, kernel_size=(2, 2))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)
    
    x=layers.Flatten()(x)
    
    x=layers.Dense(48,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(num_classes,activation="softmax")(x)
    
    return keras.Model(inputs,outputs)


num_classes = 21
input_shape = (100, 100, 1)
model=make_model(input_shape=input_shape,num_classes=num_classes)

checkpoint=tf.train.Checkpoint(model)
#checkpoint.restore(r'K:\PVP Companion ImageRec Model\Best_2.8.2022_50epochs').expect_partial()
checkpoint.restore(r'K:\PVP Companion ImageRec Model\AlternateTrain_4.18.2022_50epochs').expect_partial()

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])


#%%

#filepath=r'K:\Tessarect_ImageTest\PvP Scoreboards'

#filepath=r'K:\Tessarect_ImageTest\NewImages'

filepath=r'C:\Users\Kenneth\Box\Technical Projects\Tournament Data\Scoreboards\EU Central April Qualifier'

file_list=os.listdir(filepath)

file_extracts=[]



base_res=(3440,1440)#ask users to enter? 

for count, file in enumerate(file_list[0:10]):
    
    print(str(count)+' of '+str(len(file_list))+' files')

    img = cv2.imread(filepath+'\\'+file)
    
#    img=cv2.imread(r'K:\Tessarect_ImageTest\PvP Scoreboards\unknown (1).png')
    
    cropped=strip_bars(img)
    
    img_norm=cv2.resize(cropped,base_res,interpolation=cv2.INTER_CUBIC)
    
    ranked,match_result=detect_ranked_result(img_norm)
    if ranked==True: #if ranked detected, add the offset to constants. else keep it 0
        offset=.04
    else:
        offset=0
    
    #0 is kda, 1 is stats, 2 is class images, 3 is title bar, 4 is names
    #0 is row coords, 1 is col coords
    
    #base 21:9 images
#    consts=[
#            {'row':pd.DataFrame([0.4652777777777778,0.5347222222222222,0.6041666666666666,
#                0.6631944444444444,0.7326388888888888,0.8020833333333334]),
#             'col':pd.DataFrame([0.44825581395348835,0.4796511627906977,0.5110465116279069])},
#            {'row':pd.DataFrame([0.4552777777777778,0.5247222222222222,0.5941666666666666,
#                0.6531944444444444,0.7226388888888888,0.7920833333333334]),
#             'col':pd.DataFrame([.54])},
#            {'row':pd.DataFrame([0.455,0.515,0.575,.650,.71,.77]),
#             'col':pd.DataFrame([.2875])},
#             [.35,.05],
#             {'row':pd.DataFrame([0.4652777777777778,0.5307222222222222,0.5961666666666666,
#            0.6631944444444444,0.7266388888888888,0.7920833333333334]),
#            'col':pd.DataFrame([.31])}
#             ]
             
             
    #stream captures, with side crop (full 1920x1080)
    #0 is kda, 1 is stats, 2 is class images, 3 is title bar, 4 is names
    #0 is row coords, 1 is col coords
    
        consts=[
            {'row':pd.DataFrame([0.465,0.535,0.604,0.663,0.733,0.802]),
             'col':pd.DataFrame([0.432,0.475,0.516])},
            {'row':pd.DataFrame([0.458,0.524,0.594,0.658,0.722,0.792]),
             'col':pd.DataFrame([.55])},
             {'row':pd.DataFrame([0.455,0.515,0.575,.650,.71,.77]),
             'col':pd.DataFrame([.2175])},#.2175 original
            [.35,.05],
            {'row':pd.DataFrame([0.4652,0.5307,0.5961,0.6631,0.7266,0.7920]),
            'col':pd.DataFrame([.245])}      
            ]
    
#    also need to chance captures
             
    #stream captures, no crop
    #0 is kda, 1 is stats, 2 is class images, 3 is title bar, 4 is names
    #0 is row coords, 1 is col coords
#    consts=[
#            {'row':pd.DataFrame([0.344,0.394,0.444,.494,.544,.594]),
#             'col':pd.DataFrame([0.44825581395348835,0.4796511627906977,0.5110465116279069])},
#            {'row':pd.DataFrame([0.344,0.394,0.444,.494,.544,.594]),
#             'col':pd.DataFrame([.54])},
#            {'row':pd.DataFrame([.344,.384,.424,.474,.524,.574]),
#             'col':pd.DataFrame([.2875])},
#            [.35,.05],
#            {'row':pd.DataFrame([0.344,0.394,0.444,.494,.544,.594]),
#            'col':pd.DataFrame([.31])}      
#            ]
             
             
    #-----------------------find kda      
    
    
#    col_consts=[0.44825581395348835,0.4796511627906977,0.5110465116279069]
#    row_consts=[0.4652777777777778,0.5347222222222222,0.6041666666666666,
#                0.6631944444444444,0.7326388888888888,0.8020833333333334]
    kda_total=[]
    kda_locations=[]
    
    for e1 in consts[0]['row'][0]:
        for e2 in consts[0]['col'][0]:
            kda_locations.append([e2-offset,e1])
        
    kda_img_list=[]
    results=[]
    custom_config = r'--psm 10 -c tessedit_char_whitelist=0123456789'
    
    
    final=preProcess(img_norm)
    
    for each in kda_locations:
        x=round(each[0]*base_res[0])
        x1=round(each[0]*base_res[0])+40
        y=round(each[1]*base_res[1])
        y1=round(each[1]*base_res[1])+40
        
        sub_image=img_norm[y:y1,x:x1]
        
        kda_img_list.append(sub_image)
        
    
    for each in kda_img_list:
        
#        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#        sharpen = cv2.filter2D(each, -1, sharpen_kernel)
        
#        pre=preProcess(sharpen)
        resized=cv2.resize(each,(160,160),interpolation=cv2.INTER_CUBIC)
        
        blur=cv2.medianBlur(resized,5)
        
        gray=get_grayscale(blur)
        thresh=thresholding(gray)

 
        d=pytesseract.image_to_data(thresh, output_type=Output.DICT, config=custom_config)

        n_boxes = len(d['text'])
        
        
        found=False
        
        for i in range(n_boxes):
            if float(d['conf'][i]) >=0:
                results.append(d['text'][i])
                found=True
        
        if found==False:
            results.append('Err')

    kda_total.append(results)
        
    
    #-----------------------find battlestats
#    custom_config = r'--psm 3 outputbase digits'
    custom_config = r'--psm 12 outputbase digits'

    stats_total=[]
    
#    row_consts=[0.4552777777777778,0.5247222222222222,0.5941666666666666,
#                0.6531944444444444,0.7226388888888888,0.7920833333333334]
#    col_consts=[.54]

    stats_locations=[]
    for e1 in consts[1]['row'][0]:
        for e2 in consts[1]['col'][0]:
            stats_locations.append([e2-offset,e1])
        

    stat_img_list=[]
    results=[]
    
    for each in stats_locations:
        x=round(each[0]*base_res[0])
        x1=round(each[0]*base_res[0])+800#go to 800 for 1920x1080 conversion?
        y=round(each[1]*base_res[1])
        y1=round(each[1]*base_res[1])+55
        
        sub_image=final[y:y1,x:x1]
        
        stat_img_list.append(sub_image)
        
    
    for each in stat_img_list:
        
        d=pytesseract.image_to_data(each, output_type=Output.DICT, config=custom_config)
        
    
        n_boxes = len(d['text'])
    
        for i in range(n_boxes):
            if float(d['conf'][i]) >= 0:
                
#                print([d['conf'][i],d['text'][i]])
                
                if len(d['text'][i])>2:
                    results.append(d['text'][i])
            

    stats_total.append(results)
    
    
    
    
    #%------------------find class images and use classifier to pull image data

    class_locations=[]
    for e1 in consts[2]['row'][0]:
        for e2 in consts[2]['col'][0]:
            class_locations.append([e2-offset,e1])
    
    class_img_list=[]
    for count, each in enumerate(class_locations):
        (x,y,w,h)=(round(each[0]*base_res[0]),round(each[1]*base_res[1]),100,100)
#        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        sub=get_grayscale(img_norm[y:y+w,x:x+w])
        
        class_img_list.append(sub)
    
    
    sub_images=np.zeros((6,100,100))
    
    for count,sub in enumerate(class_img_list):
#        sub_gray=get_grayscale(sub)
    
        sub_images[count,:,:]=sub.astype("float32")/255

    x_images=np.reshape(sub_images,(6,100,100,1))
    
    predict=model.predict(x_images)
    class_results=np.argmax(predict,1)
    
    
    #----------------------find names
    
    custom_config = r'--psm 3'
    name_total=[]
    name_locations=[]
    for e1 in consts[4]['row'][0]:
        for e2 in consts[4]['col'][0]:
            name_locations.append([e2-offset,e1])
            
    name_img_list=[]
    results=[]
        
    for each in name_locations:
        x=round(each[0]*base_res[0])
        x1=round(each[0]*base_res[0])+250
        y=round(each[1]*base_res[1])
        y1=round(each[1]*base_res[1])+35
        
        sub_image=img_norm[y:y1,x:x1]
        
        name_img_list.append(sub_image)
        
    for each in name_img_list:
        
#        each=name_img_list[1]
    
        resized=cv2.resize(each,(500,70),interpolation=cv2.INTER_CUBIC)
    
#        eroded=erode(resized)
        
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(resized, -1, sharpen_kernel)
        
        gray=get_grayscale(sharpen)
        thresh=thresholding(gray)
        
        inverted=cv2.bitwise_not(thresh)
        
        d=pytesseract.image_to_data(inverted, output_type=Output.DICT, config=custom_config)
    
        found=False
        
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if float(d['conf'][i]) > 30 and len(d['text'][i])>1:
                results.append(d['text'][i])
                found=True
            
        if found==False:
            results.append('Err')
            
    name_total.append(results)

        
    item={'File':file,'Names':name_total,'Result':match_result,'KDA':kda_total,'Stats':stats_total,'Class':class_results}
    file_extracts.append(item)
    
#%%
    
    
#need to add class translation and also need to add in to the extraction area
import pandas as pd
data=pd.DataFrame()
extract_df=pd.DataFrame(data)
import pandas as pd


class_translate={0:'Berserker',1:'Destroyer',2:'Gunlancer',3:'Paladin',4:'Arcanist',5:'Summoner',
                 6:'Bard',7:'Wardancer',8:'Scrapper',9:'Soulfist',10:'Glaivier',11:'Striker',12:'Deathblade',
                 13:'Shadowhunter',14:'Reaper',15:'Sharpshooter',16:'Deadeye',17:'Artillerist',18:'Machinist',19:'Gunslinger',20:'Sorceress'}

def splitData(input_list):
    
    x = 3
    
    new_list=[]
    for i in range(0, len(input_list), x):
        new_list.append(input_list[i:i+x])
    
    return new_list
data=[]

for each in file_extracts:
    
    if len(each['KDA'][0])==18 and len(each['Stats'][0])==18:
        
        split_KDA=splitData(each['KDA'][0])
        split_stats=splitData(each['Stats'][0])
        
        for i in range(0,6):
            player_id=each['Names'][0][i]
            
            class_id=each['Class'][i]
            player_class=class_translate[class_id]
            
            kills=split_KDA[i][0]
            deaths=split_KDA[i][1]
            assists=split_KDA[i][2]
            
            dmg_dealt=split_stats[i][0]
            dmg_taken=split_stats[i][1]
            battle_pts=split_stats[i][2]
            
            item={'Game':each['File'],'Result':each['Result'],'Player':player_id,'Class':player_class,'Kills':kills,'Deaths':deaths,'Assists':assists,
                  'Damage Dealt':dmg_dealt,'Damage Taken':dmg_taken,'Battle Points':battle_pts}
            
            data.append(item)
    else:
        continue

extract_df=pd.DataFrame(data)
#%%misc, do not run
x_images=np.reshape(x_train,(2376,100,100,1))
predict=model.predict(x_images)

e1=cv2.imread(r'K:\Tessarect_ImageTest\Class Nonsense\train_img_2308.jpg')
e2=cv2.imread(r'K:\Tessarect_ImageTest\Class Nonsense\train_img_2336.jpg')
#e1=x_train[2308,:,:].img_to_array(e1, dtype='uint8')
#e2=x_train[2336,:,:]

#e1=get_grayscale(e1)
#
#e2=get_grayscale(e2)
#
#thresh1=thresholding(e1)
#thresh2=thresholding(e2)
#
#cv2.imshow('test',thresh1)
#cv2.waitKey(0)

conflist=np.zeros((predict.shape[0],1))
for i in range(predict.shape[0]):
    conflist[i,0]=np.max(predict[i,:])
    