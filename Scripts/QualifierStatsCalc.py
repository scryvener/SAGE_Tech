# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:44:58 2022

@author: Kenneth
"""

import pandas as pd
import numpy as np

def setone(num):
    if num==0:
        return 1
    else:
        return num
    
def extractClassStats(target_class,df_comp_stats_average):

    temp_list=[]
    for each in range(1,4):
        
        temp_list.append(df_comp_stats_average.query('Class'+str(each)+'=="'+target_class+'"'))
        
        
    comp_df_temp=pd.concat(temp_list)
    
    comp_df_temp['Games Won']=comp_df_temp['Win']*comp_df_temp['GamesPlayed']    
    
    return comp_df_temp



def main(region,basepath):

    for count,each in enumerate(regions):
        if count==0:
            df_compiled=pd.read_excel(basepath+each+' March Qualifier Data.xlsx',sheet_name=1)
        else:
            df_compiled=df_compiled.append(pd.read_excel(basepath+each+' March Qualifier Data.xlsx',sheet_name=1))
    
    df_compiled_kdaAdjusted=df_compiled.copy()
      
    df_compiled_kdaAdjusted['Deaths']=df_compiled['Deaths'].apply(setone)
            
    df_compiled['KDA']=(df_compiled_kdaAdjusted['Kills']+df_compiled_kdaAdjusted['Assists']/2)/df_compiled_kdaAdjusted['Deaths']
    df_compiled['DD_DT']=df_compiled['Damage Dealt']/df_compiled['Damage Taken']
    
    global_class_stat_total=df_compiled.groupby('Class').sum()
    global_class_stat_average=df_compiled.groupby('Class').mean()
    
    region_class_stat_total=df_compiled.groupby(['Class','Region']).sum()
    region_class_stat_average=df_compiled.groupby(['Class','Region']).mean().reset_index()
    
    player_class_stat_total=df_compiled.groupby(['Class','Player']).sum()
    player_class_stat_average=df_compiled.groupby(['Class','Player']).mean().reset_index()
    
    team_stat_total=df_compiled.groupby(['Team']).sum().reset_index()
    
    team_stat_total['Kills Per Win']=team_stat_total['Kills']/team_stat_total['Win']
    team_stat_total['Assists Per Win']=team_stat_total['Assists']/team_stat_total['Win']
    team_stat_total['Deaths Per Win']=team_stat_total['Deaths']/team_stat_total['Win']
    
    
    temp=df_compiled.groupby('Team').count()/3
    team_stat_total['GamesPlayed']=temp['Game'].values
    
    team_stat_total['Kills Per Game']=team_stat_total['Kills']/team_stat_total['GamesPlayed']
    team_stat_total['Assists Per Game']=team_stat_total['Assists']/team_stat_total['GamesPlayed']
    team_stat_total['Deaths Per Game']=team_stat_total['Deaths']/team_stat_total['GamesPlayed']
    
    team_stat_average=df_compiled.groupby(['Team']).mean()
    
    player_class_stat_average['DeathEff']=player_class_stat_average['Damage Taken']/player_class_stat_average['Deaths']
    player_class_stat_average['KillEff']=player_class_stat_average['Damage Dealt']/player_class_stat_average['Kills']
    
    global_class_counts=pd.DataFrame(df_compiled['Class'].value_counts()).reset_index().sort_values('index')
    global_class_counts['Per']=global_class_counts['Class']/df_compiled.shape[0]
    
    global_class_match=df_compiled[['Class','Match','Team','Region']].groupby(['Match','Team'])
    
    counts=[]
    for match,group in global_class_match:
        class_df=group[['Class','Region']]
        class_counts=class_df.drop_duplicates()
        
        class_counts['Team']=match[1]
        
        counts.append([match,class_counts])
    
    match_picks=[]
    for each in counts:
        match_picks.append(each[1])
    
    df_match_picks=pd.concat(match_picks).drop_duplicates()
    
    df_match_picks_counts=df_match_picks['Class'].value_counts()
    
    df_region_match_picks_counts=df_match_picks.value_counts().reset_index().rename(columns={0:'Counts'})
    
    
    class_list=pd.DataFrame(global_class_counts['index'])
    
    region_class_counts=pd.DataFrame(df_region_match_picks_counts[['Class','Region']].value_counts(),columns=['Counts']).reset_index()
    
    region_df=[]
    #regions=['EU Central',']
    for each in regions:
        
        df_sub=region_class_counts[['Class','Region','Counts']].query('Region=="'+each+'"')#original is region_class_counts
        
        df_sub['Per']=df_sub['Counts']/np.sum(df_sub['Counts'])
        
        region_df.append(df_sub)
        
        class_list=class_list.merge(df_sub,how='left',left_on='index',right_on='Class')
        
        del class_list['Class']
        del class_list['Region']
        
        class_list=class_list.rename(columns={'Counts':'Counts_'+each,'Per':'Per_'+each})
        
    
    class_list=class_list.fillna(0)
        
    #%pull team comps
    
    teams=df_compiled['Team'].drop_duplicates()
    comp_df=pd.DataFrame(columns=['Team','Class1','Class2','Class3'])
    for team in teams:
        sub_df=df_compiled.query('Team=="'+team+'"').sort_values('Game')
        
        classlist=sub_df[['Class','Game','Region']]
        
        games=classlist['Game'].drop_duplicates()
        
        
        for game in games:
            game_df=classlist.query('Game=="'+game+'"').sort_values('Class')
            
            comp=game_df['Class']
            
            item={'Team':team,'Game':game,'Region':game_df['Region'].iloc[0],'Class1':comp.iloc[0],'Class2':comp.iloc[1],'Class3':comp.iloc[2],}
            
            comp_df=comp_df.append(item,ignore_index=True)
        
            
    comp_pick_rate_df=comp_df.groupby(['Class1','Class2','Class3','Region','Team']).count().reset_index()       
    global_comp_pick_rate_df=comp_df.groupby(['Class1','Class2','Class3','Team']).count().reset_index()
    
    global_comp_pick_rate_collapsed=global_comp_pick_rate_df.groupby(['Class1','Class2','Class3']).count().reset_index()
    
    general_comp_pick_rate=comp_df.groupby(['Team']).count().reset_index()
    
    df_team_game=df_compiled.groupby(['Game','Team']).mean().reset_index()
    comp_stats=pd.merge(comp_df,df_team_game,how='inner',left_on=['Game','Team'],right_on=['Game','Team'])
    
    temp2=comp_stats.groupby(['Class1','Class2','Class3']).count().reset_index()
    
    df_comp_stats_average=comp_stats.groupby(['Class1','Class2','Class3']).mean().reset_index()
    
    df_comp_stats_average['GamesPlayed']=temp2['Game']
    
    temp3=comp_stats.groupby(['Class1','Class2','Class3','Region']).count().reset_index()
    df_comp_stats_average_region=comp_stats.groupby(['Class1','Class2','Class3','Region']).mean().reset_index()
    df_comp_stats_average_region['GamesPlayed']=temp3['Game']
    
    df_bard_comp_stats=extractClassStats('Bard',df_comp_stats_average)
    bard_comp_wr=np.sum(df_bard_comp_stats['Games Won'])/np.sum(df_bard_comp_stats['GamesPlayed'])
    
    df_paladin_comp_stats=extractClassStats('Paladin',df_comp_stats_average)
    paladin_comp_wr=np.sum(df_paladin_comp_stats['Games Won'])/np.sum(df_paladin_comp_stats['GamesPlayed'])
    
    df_gunlancer_comp_stats=extractClassStats('Gunlancer',df_comp_stats_average)
    gunlancer_comp_wr=np.sum(df_gunlancer_comp_stats['Games Won'])/np.sum(df_gunlancer_comp_stats['GamesPlayed'])
    
    
    df_soulfist_comp_stats=extractClassStats('Soulfist',df_comp_stats_average)
    soulfist_comp_wr=np.sum(df_gunlancer_comp_stats['Games Won'])/np.sum(df_gunlancer_comp_stats['GamesPlayed'])
    
    
    print([bard_comp_wr,paladin_comp_wr,gunlancer_comp_wr])


if __name__ == "__main__":
    
    regions=['EU Central','NA East','NA West','SA']
    
    basepath=r'C:\Users\Kenneth\Box\Technical Projects\Tournament Data\Scoreboards\\'
    
    main()



