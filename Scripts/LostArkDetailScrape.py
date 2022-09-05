# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 19:25:48 2021

@author: Kenneth
"""

#%%extract skill descriptions 
    
import requests
import time
import random

skill_lists=[]
id_list=['102','103','104','105','202','203','204','302','303','304','305','312','402','403','404','502','503','504','505','512']

for class_id in id_list:
    
    print('Working on '+class_id)

    x=requests.post(r'https://lostarkcodex.com/ajax.php?1=en',data={'a': 'skill_list','class_id':class_id,'lang':'us'})
    
    skill_json=x.json()
    
    skill_lists.append(skill_json)
    
    time.sleep(round(random.random()*1000))

import pickle

pickle.dump(skill_lists,open(r'K:\SkillData.pkl','wb'))


#%quick measure, comment out later

#keylist=[]
#for each in skill_lists:
#    data=each['data']
#    skills=data['skills']
#    
#    keys=skills.keys()
#
#    keylist.extend(keys)


#%%scrape skill details

import demjson
import re

error_count=0

saved_descriptions=[]

for each in skill_lists:
    data=each['data']
    
    skills=data['skills']
    
    keys=skills.keys()
    
    for key in keys:
        
        data_dict=skills[key]
        path=r'https://lostarkcodex.com/us/skill/'+key+'/'
        
        a=requests.get(path)
        
        if a.status_code==200:
            print('Pulled Skill Data for: '+key)
        
            temp=a.text
            
            temp2=re.search('<script>.*</script>',temp)
            
            detail=temp2.group()
            
            cleaned=detail.replace('<script>var skill_stats = ','')
            cleaned=cleaned.replace(';</script>','')
            
            py_data=demjson.decode(cleaned)
    
            for i in py_data.keys():
                
                if i=='max_level':
                    continue
                else:
                
                    data=py_data[i].copy()
                    
                    sdesc=data['sdesc']
                
                    clean_descrip=re.sub('<.*?>','',sdesc)
                    
                    py_data[i]['sdesc']=clean_descrip
                    
            data_dict['Descriptions']=py_data
            
            saved_descriptions.append([key,py_data])
            
            time.sleep(10+round(random.random()*30))
            
        elif a.status_code != 200 and error_count<10:
            error_count=error_count+1
            print('Error: '+r.status_code+' Site could not be succesfully retrieved, sleeping...')
            time.sleep(round(300+random.random()*600))
            
        else:
            print('Error limit exceeded, terminating scrape')
            break


#%%pull image files we need

skill_img_list=[]
#trip_img_list=[]

for each in extra_skill_list:#change based on what skilllist you are using
#    data=each['data']
#    
#    skills=data['skills']
#    
#    keys=skills.keys()
    
#    for key in keys:
#        
#        data_dict=skills[key]
    
    url=each['icon']
    clean_url=url.replace('"','')
    
        
    skill_img=r'https://lostarkcodex.com'+clean_url
    
    skill_img_list.append(skill_img)
    
    each['icon']=clean_url
    
#        tripods=data_dict['e']
#        
#        tripod_keys=tripods.keys()
#        
#        for tri_key in tripod_keys:
#            
#            tripod_dict=tripods[tri_key]
#            
#            tripod_img=r'https://lostarkcodex.com/icons/'+tripod_dict['icon']
#            
#            trip_img_list.append(tripod_img)
#        

skill_img_df=pd.DataFrame(skill_img_list,columns=['URL']).drop_duplicates()
#trip_img_df=pd.DataFrame(trip_img_list,columns=['URL']).drop_duplicates()        

#%sample code
#for i in range(0,5):
#    
#    r = requests.get(url, stream = True)
#    
#    r.raw.decode_content = True
#            
#    # Open a local file with wb ( write binary ) permission.
#    with open(filename,'wb') as f:
#        shutil.copyfileobj(r.raw, f)
#        
#    time.sleep(15)

#%%begin scraping

## Importing Necessary Modules
import requests # to get image from the web
import shutil # to save it locally


error_count=0
for url in skill_img_df['URL']:

    filename = "K:\LostArkIcons\\" +url.split("/")[-1]

# Open the url image, set stream to True, this will return the stream content.
    r = requests.get(url, stream = True)
    
# Check if the image was retrieved successfully

    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',url.split("/")[-1])
        time.sleep(60+round(random.random()*30))
        
    elif r.status_code != 200 and error_count<10:
        error_count=error_count+1
        print('Error: '+r.status_code+' Image could not be succesfully retrieved, sleeping...')
        time.sleep(round(300+random.random()*600))
        
    else:
        print('Error limit exceeded, terminating scrape')
        break
    
#%% also need to clean the tripod descriptions
from xml.etree import ElementTree


class_list=[['Berserker',102],['Destroyer',103],['Gunlancer', 104],['Paladin',105],['Arcanist',202],['Summoner',203],['Bard',204],['Wardancer',302],
            ['Scrapper',303], ['Soulfist',304],['Glaiver',305],['Striker',312],['Deathblade',402],['Shadowhunter',403],['Reaper',404],['Sharpshooter',502],
            ['Deadeye',503],['Artillerist',504],['Machinist',505],['Gunslinger',512]
]



trip_desc=[]
for count,each in enumerate(skill_lists):
    data=each['data']
    
    skills=data['skills']
    
    keys=skills.keys()
    
    for key in keys:
        
        data_dict=skills[key]
        
        for i in range(1,len(data_dict['e'])+1):
            tripod=data_dict['e'][str(i)]
            
            clean_list=[]
            
            cleaned_test=re.sub('(<FONT COLOR.*?>|</FONT>)','',tripod['desc'].replace('&gt;','>').replace('&lt;','<').replace('&quot;','"').replace('<br>',''))
            
            test_tree=ElementTree.fromstring(cleaned_test)
            
            target_divs=test_tree.findall(".//div[@class='mb-1']")
            
            for div in target_divs:
                if div.text is not None:
                    clean_list.append(div.text)
                else:
                    continue
                
            target_descrip=clean_list[-1]
            
            trip_parent=class_list[count][0]
            trip_parent_id=class_list[count][1]
            
            trip_name=tripod['name']
            trip_icon=tripod['icon']
            
            trip_parent_skill=key
            
            item={'name':trip_name,'icon':trip_icon,'ParentClass':trip_parent,'ParentID':trip_parent_id,'ParentSkill':trip_parent_skill,'Description':target_descrip}
            trip_desc.append(item)
            
            
            
trip_df=pd.DataFrame(trip_desc)
            
pickle.dump(trip_df,open('K:\TripodDescriptions.pkl','wb'))
            
#skill_lists[18]['data']['skills']['35100']['e'][str(7)]['desc']="&lt;div class=&quot;skill_feature_tooltip_container&quot;&gt;&lt;div class=&quot;skill_feature_tooltip&quot;&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;All-out War&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;&lt;FONT COLOR='#3C78FF'&gt;PvE&lt;/FONT&gt;&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Lv. 1&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;Max Level 5&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;mb-1&quot;&gt;No longer consumes battery on skill use. Consumes &lt;FONT COLOR='#ff9999'&gt;100&lt;/FONT&gt; battery resources per self-destructing drone. Damage +&lt;FONT COLOR='#6fffcf'&gt;120.0%&lt;/FONT&gt;. Skill canceled when remaining battery resources are below &lt;FONT COLOR='#ffff99'&gt;100&lt;/FONT&gt;.&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;All-out War&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;&lt;FONT COLOR='#3C78FF'&gt;PvE&lt;/FONT&gt;&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Lv. 5&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;Max Level 5&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;mb-1&quot;&gt;No longer consumes battery on skill use. Consumes &lt;FONT COLOR='#ff9999'&gt;100&lt;/FONT&gt; battery resources per self-destructing drone. Damage +&lt;FONT COLOR='#6fffcf'&gt;170.0%&lt;/FONT&gt;. Skill canceled when remaining battery resources are below &lt;FONT COLOR='#ffff99'&gt;100&lt;/FONT&gt;.&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;skill_feature_tooltip ml-1&quot;&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;All-out War&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;&lt;FONT COLOR='#FF4646'&gt;PvP&lt;/FONT&gt;&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Lv. 1&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;Max Level 5&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;mb-1&quot;&gt;No longer consumes battery on skill use. Consumes &lt;FONT COLOR='#ff9999'&gt;100&lt;/FONT&gt; battery resources per self-destructing drone. Damage +&lt;FONT COLOR='#6fffcf'&gt;120.0%&lt;/FONT&gt;. Skill canceled when remaining battery resources are below &lt;FONT COLOR='#ffff99'&gt;100&lt;/FONT&gt;.&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;All-out War&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;&lt;FONT COLOR='#FF4646'&gt;PvP&lt;/FONT&gt;&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Lv. 5&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;Max Level 5&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;mb-1&quot;&gt;No longer consumes battery on skill use. Consumes &lt;FONT COLOR='#ff9999'&gt;100&lt;/FONT&gt; battery resources per self-destructing drone. Damage +&lt;FONT COLOR='#6fffcf'&gt;170.0%&lt;/FONT&gt;. Skill canceled when remaining battery resources are below &lt;FONT COLOR='#ffff99'&gt;100&lt;/FONT&gt;.&lt;/div&gt;&lt;/div&gt;&lt;/div&gt;"
#skill_lists[count]['data']['skills']['35090']['e'][str(i)]['desc']="&lt;div class=&quot;skill_feature_tooltip_container&quot;&gt;&lt;div class=&quot;skill_feature_tooltip&quot;&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Pulse Stack&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;&lt;FONT COLOR='#3C78FF'&gt;PvE&lt;/FONT&gt;&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Lv. 1&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;Max Level 1&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;mb-1&quot;&gt;The skill can now stack up to &lt;FONT COLOR='#ffff99'&gt;2&lt;/FONT&gt;.&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;skill_feature_tooltip ml-1&quot;&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Pulse Stack&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;&lt;FONT COLOR='#FF4646'&gt;PvP&lt;/FONT&gt;&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;clearfix&quot;&gt;&lt;div class=&quot;float-left&quot;&gt;Lv. 1&lt;/div&gt;&lt;div class=&quot;float-right&quot;&gt;Max Level 1&lt;/div&gt;&lt;/div&gt;&lt;div class=&quot;mb-1&quot;&gt;&lt;/div&gt;&lt;/div&gt;&lt;/div&gt;"
     
#%%clean and ready skill dicts

from dict2xml import dict2xml

class_list=[['Berserker',102],['Destroyer',103],['Gunlancer', 104],['Paladin',105],['Arcanist',202],['Summoner',203],['Bard',204],['Wardancer',302],
            ['Scrapper',303], ['Soulfist',304],['Glaiver',305],['Striker',312],['Deathblade',402],['Shadowhunter',403],['Reaper',404],['Sharpshooter',502],
            ['Deadeye',503],['Artillerist',504],['Machinist',505],['Gunslinger',512]
]

cleaned_list={}
        
for count,each in enumerate(skill_lists):
    temp=each['data']['skills']
    
    
    class_dict={'Name':class_list[count][0],'ClassID':class_list[count][1],'Skills':temp}
    
    
    temp_xml=dict2xml(class_dict)
    
    cleaned_list[class_list[count][0]]=temp_xml
    
    
#%%find existing skills and compare to what we need to scrape

existing=[]
for each in skill_lists:
    temp=list(each['data']['skills'].keys())    
    
    existing.extend(temp)

        
overall=pd.read_excel(r'C:\Users\Kenneth\Box\Technical Projects\Overlay\Skill_IDs.xlsx',dtype=str)

#check and remove the skills that are in existing already
remaining=[]
for count,each in enumerate(overall['SkillID']):
    
    if each not in existing:
        remaining.append(overall.iloc[count])
    else:
        continue

remaining_df=pd.DataFrame(remaining)

#%%alternate scrape for pulling img urls and also tripod less skills
import demjson
import re

error_count=0

#extra_skill_list=[]
#responses=[]

pickle.dump(extra_skill_list,open('K:\ExtraSkills.pkl','wb'))
pickle.dump(responses,open('K:\ExtraSkills_webresponse.pkl','wb'))

count=79
for count,skillID in enumerate(remaining_df['SkillID'].iloc[79:]):
    
    
    path=r'https://lostarkcodex.com/us/skill/'+skillID+'/'
    
    a=requests.get(path)
    
    
    if a.status_code==200:
        print('Pulled Skill Data for: '+skillID)
    
        temp=a.text
        
        responses.append(a)
        
        temp2=re.search('<script>.*</script>',temp)

        detail=temp2.group()
            
        cleaned=detail.replace('<script>var skill_stats = ','')
        cleaned=cleaned.replace(';</script>','')
        
        py_data=demjson.decode(cleaned)
        
        if 'sdesc' not in py_data['1'].keys():
            temp2=re.search('<div id="stat-sdesc">.*</div>',temp)
            
            detail=temp2.group()
            
            cleaned=detail.replace('<div id="stat-sdesc">','')
            cleaned=cleaned.replace('</div>','')
            
        py_data['1']['sdesc']=cleaned
        
        
#        remaining_df['Description'].iloc[count]=cleaned
    
        temp_img=re.search('"\/icons.*.png"',temp)
        
        temp_name=re.search('"name": "(.*)"',temp)
        
        name_detail=temp_name.group()
        
        name_detail=name_detail.replace('"name": "','').replace('"','')
        
        img_detail=temp_img.group()
        
        
        item={"skillID":skillID,'name':name_detail,'Descriptions':py_data['1']['sdesc'],'icon':img_detail}
        extra_skill_list.append(item)
        
        time.sleep(30+round(random.random()*30))
        
    elif a.status_code != 200 and error_count<10:
            error_count=error_count+1
            print('Error: '+r.status_code+' Site could not be succesfully retrieved, sleeping...')
            time.sleep(round(300+random.random()*600))
            
    else:
        print('Error limit exceeded, terminating scrape')
        break
            
#%%clean extraskill descs


orig_extra=pickle.load(open('K:\ExtraSkills.pkl','rb'))
   
for each in extra_skill_list:
    
    data=each.copy()
                    
    sdesc=data['Descriptions']

    clean_descrip=re.sub('<.*?>','',sdesc)
    
    each['Descriptions']=clean_descrip
    
    

#%% create total skill list

#use pve descriptions if no pvp tag available, only descrips up to lv 10. 

total_skills_list=[]

for each in skill_lists:
    
    data=each['data']['skills']
    
    skill_keys=data.keys()
    
    for key in skill_keys:
        
        skillID=key
        name=data[key]['name']
        icon=data[key]['icon']
        
        target_keys=list(range(1,11))
        descriptlist=data[key]['Descriptions']
        
        descriptions={}
        
        for target in target_keys:
            descriptions[str(target)]=descriptlist[str(target)]['sdesc']
        
        item={'skillID':skillID,'name':name,'icon':icon,'Descriptions':descriptions}
        
        total_skills_list.append(item)
        
total_skills_list.extend(extra_skill_list[0:80])

total_skills_list.extend(temp_extra)
                
#%clean titles of all the skills

for each in total_skills_list:
    temp_name=each['name']
    
    final=temp_name.replace('&amp;#39;',"'")
    
    each['name']=final
    
pickle.dump(total_skills_list,open(r'K:\TotalSKillData.pkl','wb'))

#%fix icon path

total_skills_list=pickle.load(open(r'K:\TotalSKillData.pkl','rb'))

for each in total_skills_list:
    
    splitpath=each['icon'].split("/")
    
    newpath="LostArkIcons\\"+splitpath[-1]
    
    each['icon']=newpath
pickle.dump(total_skills_list,open(r'K:\TotalSKillData.pkl','wb'))


trip_df['icon']="LostArkIcons\\"+trip_df['icon']

pickle.dump(trip_df,open('K:\TripodDescriptions.pkl','wb'))




    