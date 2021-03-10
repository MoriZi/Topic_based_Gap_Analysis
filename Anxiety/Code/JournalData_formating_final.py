import pandas as pd

df_BRT = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Behavior Research and Therapy_1-631_cleaned.csv')
df_BMC = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/BMC Psychiatry_1-699_cleaned.csv')
df_FPsycho = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/FPsychology.csv')
df_FPsychi = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Frontiers in Psychiatry_1-653_cleaned.csv')
df_JAnD = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/J anxiety disorders_1-642_cleaned.csv')
df_CCP = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/J of Counselling and Clininal Psy_1-298_cleaned.csv')
df_JAD_1 = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/JAD_01.csv')
df_JAD_2 = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/JAD_02.csv')
df_JAD_3 = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/JAD_03.csv')
df_PONE_1 = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/PONE_01.csv')
df_PONE_2 = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/PONE_02.csv')
df_PR = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/PResearch.csv')
df_PM = pd.read_csv('./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Psychological Medicine_1-628_cleaned.csv')

frames=[df_BRT,df_BMC,df_FPsycho,df_FPsychi,df_JAnD,df_CCP,df_JAD_1,df_JAD_2,df_JAD_3,df_PONE_1,df_PONE_2,df_PR,df_PM]
df=pd.concat(frames)
#df.head()

df11=df[df['year']==2011]
#df11.head()
df11.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2011.csv")
df12=df[df['year']==2012]
df12.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2012.csv")
df13=df[df['year']==2013]
df13.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2013.csv")
df14=df[df['year']==2014]
df14.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2014.csv")
df15=df[df['year']==2015]
df15.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2015.csv")
df16=df[df['year']==2016]
df16.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2016.csv")
df17=df[df['year']==2017]
df17.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2017.csv")
df18=df[df['year']==2018]
df18.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2018.csv")
df19=df[df['year']==2019]
df19.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2019.csv")
df20=df[df['year']==2020]
df20.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2020.csv")
df21=df[df['year']==2021]
df21.to_csv("./Topic_based_Gap_Analysis/Anxiety/Data/CSV/Acad_2021.csv")