# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:22:04 2017

@author: shrad
"""

import os
import pandas as pd

os.chdir('C:\Users\shrad\Desktop\Recommendation_Systems\MatrixFactorization')

movies = pd.read_csv('movies.csv')
rec_df = pd.read_csv('all_users_rec.csv')

user_id = raw_input("Enter user id for recommendation:")
user_id = int(user_id)

outfile = open("Recommendations_"+str(user_id)+".txt","w")

sim_list = rec_df.loc[rec_df['userId']==user_id]['rec_movieId']
sim_list = list(sim_list)

for item in sim_list:
    text=movies.loc[movies['movieId']==item].iloc[:,1:4]
    outfile.write(str(text))
    
outfile.close()

print "Generated Recommendations"
