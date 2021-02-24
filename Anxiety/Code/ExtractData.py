#Importing necessary libraries
import json
import os
import sys
import pandas as pd
import praw
from datetime import datetime
import time
import pytz

def extract_links(fname):
    '''
    fname: JSON file name
    This function is used to extract the links of Reddit posts in a JSON file
    '''
    links=[]
    f=open(fname)
    # returns JSON object as a dictionary
    data = json.load(f)
    # Iterating through the json
    for i in data['data']:
        links.append(i['full_link'])
    f.close()
    return links

#Function to connvert unix time to EST
def convert_to_est(utc_date):
    '''
	Given a date and time in UTC, return the date and time converted to EST
	'''
    datetime_obj = datetime.strptime(utc_date, "%Y-%m-%d %H:%M:%S")
    tz = pytz.timezone('US/Eastern')
    est_time = pytz.utc.localize(datetime_obj, is_dst=None).astimezone(tz)
    est_time_final = est_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    return est_time_final

def extract_info(post_url,df):
    '''
    post_url: Link to a given reddit post
    Given a link to a reddit post, retrieve the title, content, number of comments and date it was posted.
    '''
    #Create a reddit instance
    reddit = praw.Reddit(client_id='Vi6O78mbrmDgXw', client_secret='GilEw-IdfrLMAG30Gl5nvfsp-3bTqg', user_agent='AnxietyGA')
    #Access desired post details
    submission = reddit.submission(url=post_url)
    title=submission.title
    print(submission.title)
    text=submission.selftext
    print(submission.selftext)
    ncoms=submission.num_comments
    print(submission.num_comments)
    #Converting the submission's post date to EST
    utime=submission.created
    converted_date = convert_to_est(datetime.utcfromtimestamp(utime).strftime('%Y-%m-%d %H:%M:%S'))
    print(converted_date)
    title.replace(',','')
    text.replace(',','')
    df.loc[len(df.index)] = [title, text, ncoms,converted_date]
    return df

def main():
    '''
    Takes the name of the JSON file (or filepath) as first parameter on command line
    '''
    fname=sys.argv[1]
    df=pd.DataFrame(columns=['Title','Text','Number of Comments','Date Posted'])
    flinks=extract_links(fname)
    for link in flinks:
        extract_info(link,df)
    df.to_csv("rAnxiety2011.csv")
    return df

if __name__ == "__main__":
    main()
