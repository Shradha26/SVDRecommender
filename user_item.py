# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:57:58 2017

@author: shrad
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from math import sqrt

os.chdir('C:\Users\shrad\Desktop\Recommendation_Systems\MatrixFactorization')

ratings = pd.read_csv('ratings.csv')
                                     
user_item_mat = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
UI=user_item_mat.as_matrix()
ui_df = pd.DataFrame(UI,index=user_item_mat.index,columns=user_item_mat.columns)
#ui_df.to_csv('uimat.csv',sep=',')
user_ratings_mean=np.mean(UI,axis=0)
user_ratings_mean=user_ratings_mean.reshape(1,-1)
UI_demeaned=UI-user_ratings_mean

U,sigma,Vt=svds(UI_demeaned,150)
sigma=np.diag(sigma)

pred_mat=np.dot(np.dot(U,sigma),Vt) + user_ratings_mean
               
def calc_RMSE(actual,predicted):
    count=0
    err_sum=0
    
    for u in range(0,671):#there are 671 users but this can also be dynamically allocated
        #print u
        nz_ind=list(np.nonzero(actual[u])[0])
        for m in nz_ind:               
            err_sum+= (actual[u][m]-predicted[u][m])**2
            count+=1

    rmse=sqrt(float(err_sum)/float(count)) 
    return rmse

#print calc_RMSE(UI,pred_mat)

prediction_df=pd.DataFrame(pred_mat,index=user_item_mat.index,columns=user_item_mat.columns)
#prediction_df.to_csv('prediction.csv',sep=',')

def recommend_movies(rating_df,pred_df,n):
    
    index=0
    sim_df=pd.DataFrame(index=range(671*n+1),columns=['userId','rec_movieId','pred_rating'])
    for u in range(1,672):#range of userIds; can be specified dynamically
        #print u
            
        temp1 = list(ratings.loc[ratings['userId']==u].iloc[:,1])
        temp2=pred_df.loc[u].sort_values(ascending=False,inplace=False)
        
        count=0
        for m in temp2.index.values:
            if count==n :
                break
            if m in temp1:
                continue
            sim_df.loc[index,'userId']=u
            sim_df.loc[index,'rec_movieId']=m
            sim_df.loc[index,'pred_rating']=pred_df.loc[u,m]
            
            index+=1
            count+=1
            
      
                       
    sim_df=sim_df.dropna()    
    sim_df.to_csv("all_users_rec.csv",sep=',')
    
recommend_movies(ratings,prediction_df,15)
