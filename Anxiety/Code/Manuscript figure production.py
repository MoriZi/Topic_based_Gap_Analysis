import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('~/PycharmProjects/GapAnalysis/Topic_based_Gap_Analysis/Anxiety/Code/apa.mplstyle') # selecting the style sheet

#plot for perplexity
year = [2012,2013,2014,2015,2016,2017,2018,2019,2020]
#year = np.arange(2012,2020,1)
Pe_Aca = [-10.128,-9.637,-8.376,-7.376,-7.849,-7.784,-6.616,-6.596,-7.393]
Pe_Med = [-6.98,-7.60,-7.93,-8.96,-9.46,-6.854,-7.67,-6.854,-7.077]
Pe_Red = [-7.380,-7.201,-7.260,-6.975,-6.698,-6.686,-7.856,-5.896,-7.906]
Co_Aca = [0.371,0.365,0.374,0.362,0.342,0.330,0.356,0.351,0.353]
Co_Med = [0.56,0.35,0.39,0.33,0.44,0.43,0.37,0.43,0.45]
Co_Red = [0.290,0.295,0.294,0.303,0.337,0.329,0.290,0.347,0.303]

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