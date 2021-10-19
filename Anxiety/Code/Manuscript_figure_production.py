import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('~/PycharmProjects/GapAnalysis/Topic_based_Gap_Analysis/Anxiety/Code/apa.mplstyle') # selecting the style sheet

#data input
year = [2012,2013,2014,2015,2016,2017,2018,2019,2020]
#Perplexity
Pe_Aca = [-10.128,-9.637,-8.376,-7.376,-7.849,-7.784,-6.616,-6.596,-7.393]
Pe_Med = [-6.98,-7.60,-7.93,-8.96,-9.46,-6.854,-7.67,-6.854,-7.077]
Pe_Red = [-7.380,-7.201,-7.260,-6.975,-6.698,-6.686,-7.856,-5.896,-7.906]
#Coherence
Co_Aca = [0.371,0.365,0.374,0.362,0.342,0.330,0.356,0.351,0.353]
Co_Med = [0.56,0.35,0.39,0.33,0.44,0.43,0.37,0.43,0.45]
Co_Red = [0.290,0.295,0.294,0.303,0.337,0.329,0.290,0.347,0.303]
#Dimension composition of the biopsychsocial model
df_Aca = pd.read_csv('~/PycharmProjects/GapAnalysis/Topic_based_Gap_Analysis/Anxiety/Data/CSV/Dominant_Topics_journals.csv')
df_Med = pd.read_csv('~/PycharmProjects/GapAnalysis/Topic_based_Gap_Analysis/Anxiety/Data/CSV/Dominant_Topics_medium.csv')
df_Red = pd.read_csv('~/PycharmProjects/GapAnalysis/Topic_based_Gap_Analysis/Anxiety/Data/CSV/Dominant_Topics_Reddit.csv')

#Plot perplexity
plt.figure()
plt.plot(year, Pe_Aca, label=r'Academic Journals')
plt.plot(year, Pe_Med, label = r'Medium')
plt.plot(year, Pe_Red, label = r'Reddit')
plt.xlabel('Year')
plt.ylabel('Perplexity')
plt.legend()
plt.xlim([2012,2020])
plt.ylim([-11,0])
plt.show()

#Plot coherence
plt.figure()
plt.plot(year, Co_Aca, label=r'Academic Journals')
plt.plot(year, Co_Med, label = r'Medium')
plt.plot(year, Co_Red, label = r'Reddit')
plt.xlabel('Year')
plt.ylabel('Coherence')
plt.legend()
plt.xlim([2012,2020])
plt.ylim([0,0.6])
plt.show()

#Plot the composition of the three dimentions of the biopsychosocial model for platforms
#data process
chart_df_aca = df_Aca.groupby(['year']).agg({'year':"mean",'bio_score':"mean",'psych_score':"mean",'social_score':"mean"})
chart_df_aca['bio_score'] = chart_df_aca['bio_score']*10
chart_df_aca['psych_score'] = chart_df_aca['psych_score']*10
chart_df_aca['social_score'] = chart_df_aca['social_score']*10
chart_df_med = df_Med.groupby(['year']).agg({'year':"mean",'bio_score':"mean",'psych_score':"mean",'social_score':"mean"})
chart_df_med['bio_score'] = chart_df_med['bio_score']*10
chart_df_med['psych_score'] = chart_df_med['psych_score']*10
chart_df_med['social_score'] = chart_df_med['social_score']*10
chart_df_red = df_Red.groupby(['year']).agg({'year':"mean",'bio_score':"mean",'psych_score':"mean",'social_score':"mean"})
chart_df_red['bio_score'] = chart_df_red['bio_score']*10
chart_df_red['psych_score'] = chart_df_red['psych_score']*10
chart_df_red['social_score'] = chart_df_red['social_score']*10

#plot Academia
plt.figure()
plt.plot(chart_df_aca['year'],chart_df_aca['bio_score'],label=r'Biological')
plt.plot(chart_df_aca['year'],chart_df_aca['psych_score'],label=r'Psychological')
plt.plot(chart_df_aca['year'],chart_df_aca['social_score'],label=r'Social')
plt.xlabel('Year')
plt.ylabel('Dimension composition in percentage (%)')
plt.legend()
plt.xlim([2012,2020])
plt.ylim([0,100])
plt.show()

#plot medium
plt.figure()
plt.plot(chart_df_med['year'],chart_df_med['bio_score'],label=r'Biological')
plt.plot(chart_df_med['year'],chart_df_med['psych_score'],label=r'Psychological')
plt.plot(chart_df_med['year'],chart_df_med['social_score'],label=r'Social')
plt.xlabel('Year')
plt.ylabel('Dimension composition in percentage (%)')
plt.legend()
plt.xlim([2012,2020])
plt.ylim([0,100])
plt.show()

#plot Reddit
plt.figure()
plt.plot(chart_df_red['year'],chart_df_red['bio_score'],label=r'Biological')
plt.plot(chart_df_red['year'],chart_df_red['psych_score'],label=r'Psychological')
plt.plot(chart_df_red['year'],chart_df_red['social_score'],label=r'Social')
plt.xlabel('Year')
plt.ylabel('Dimension composition in percentage (%)')
plt.legend()
plt.xlim([2012,2020])
plt.ylim([0,100])
plt.show()