#Import Necessary Libraries
import praw
from datetime import datetime
import time
import pytz

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

#Create a reddit instance after creating a script app on the reddit website
#Link to create a script app: 
reddit = praw.Reddit(client_id='Vi6O78mbrmDgXw', client_secret='GilEw-IdfrLMAG30Gl5nvfsp-3bTqg', user_agent='AnxietyGA')

#Retrieving a post and its essential details via its URL
submission = reddit.submission(url="https://www.reddit.com/r/Anxiety/comments/15ete8/i_recently_explained_to_my_girlfriend_of_3_years/")
print(submission.title)
print(submission.selftext)
print(submission.num_comments)

#Converting the submission's post date to EST
utime=submission.created
converted_date = convert_to_est(datetime.utcfromtimestamp(utime).strftime('%Y-%m-%d %H:%M:%S'))
print(converted_date)
