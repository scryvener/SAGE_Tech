# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:31:59 2022

@author: Kenneth
"""

#pull data from smash.gg api
import requests
import time
import pickle
import pandas as pd


def main(basepath,query,events):

    data_list=[]
    for event in events:
        variables='{"eventID": '+str(event)+'}'
        response=requests.post(basepath,data={"query":query,"variables":variables},headers={"Authorization": "Bearer a2f505b7a291e5ebc0e4c834bba78b4a"})
        
        data_list.append(response.json())
        time.sleep(1)
    #    print(response.status_code)
    
    pickle.dump(data_list,open(r'C:\Users\Kenneth\Box\Technical Projects\Tournament Data\MarchSkirmishEventResults.pkl','wb'))

    #% analysis

    for count, each in enumerate(data_list):
        
        name=each['data']['event']['name']
        
        eventNodes=each['data']['event']['sets']['nodes']
        
        df=pd.DataFrame(eventNodes)
        
        df['Event']=name
        
        df['MinutesTaken']=(df['completedAt']-df['startedAt'])/60
        
        games_played=[]
        
        for each in eventNodes:
            
            if each['games']!=None:
                games=len(each['games'])
                games_played.append(games)
            else:
                games_played.append(0)
        
        df['GamesPlayed']=games_played
        
        if count==0:
            df_final=df
        else:
            df_final=df_final.append(df)
                
    df_final.to_csv(r"C:\Users\Kenneth\Box\Technical Projects\Tournament Data\MarchSkirmishEventResults.csv")

if __name__ == "__main__":
    basepath=r'https://api.smash.gg/gql/alpha'

    query="query EventStandings($eventID: ID!) {\
      event(id:$eventID){\
        name\
        sets(page:1, perPage:250){\
          nodes {\
            id\
            startedAt\
            completedAt\
            games{id}\
          }\
        }\
      }\
    }"
    
    events=[685717,685719,685720,685721]#manually enter this
    
    main(basepath,query,events)
    
