import urllib.request, urllib.error, urllib.parse
import sys

def DownloadJSON(after,month_year):
    '''
    after=the starting date in unix time
    month_year=month and year queried (example:Jan2011)
    '''
    #One Week is this many seconds in unix time
    week=604800
    for i in range(0,4):
        #before: date of one week after the start date
        before=int(after)+week
        url='https://api.pushshift.io/reddit/submission/search/?after=' + str(after) + '&before=' + str(before) + '&sort_type=score&sort=desc&subreddit=anxiety'
        print(url)
        #Extracting the contents of the JSON file of that week
        response = urllib.request.urlopen(url)
        webContent = response.read()
        print('\n')
        #Saving the data to a local JSON file.
        f = open(month_year+'w' + str(i+1)+ '.json', 'wb')
        f.write(webContent)
        f.close
        after=before

def main():
    after=sys.argv[1]
    month_year=sys.argv[2]
    DownloadJSON(after,month_year)

if __name__ == "__main__":
    main()
