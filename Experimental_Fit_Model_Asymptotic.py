
#import pandas as pd
#from os import listdir
#from os.path import isfile, join
#from datetime import datetime
#import csv
import matplotlib.pyplot as plt
import numpy as np
#import math
#import time
from scipy import optimize
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D

import sdeint
#from scipy import integrate
from scipy.optimize import curve_fit
from scipy.special import lambertw

#Constants
alpha = 0.5 #In months^{-1}
k = 0.05 #In months^{-1}
gamma = 1

def v_asymptotic_params(tvalue,tshift,mu):
    x=mu*np.exp(gamma*k*(tvalue-tshift*np.ones(np.size(tvalue))))
    #Solving equation z e^z = x
    z=lambertw(x)
    #we only have information about |v|, choosing minus
    v=-(z/mu)**(1/gamma)
    return np.real_if_close(v)

def f_SDE(x, t):
    v = x[0]
    e = x[1]
    #Function g is negative to address Referee's concerns
    g_val = - k * np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt = -alpha*(v-e)
    dedt = g_val
    return np.array([dvdt, dedt])

def sigma_SDE(x, t):
    return B


##Estimates from Wheatcroft and Davies, 1994, Wheat production in USSR, 1913-1940 (Mill Tons)
#Year; Soviet estimate reported; Soviet Estimate Revised; Low Western Estimate
Years=np.arange(1928,1941,1)
t0 =np.min(Years)
Years_All = np.arange(1924,1941,1) #Stalin comes to power at 1924
Soviet_initial_est=np.array([73.3, 71.7, 83.5, 69.5, 69.8, 89.8, 89.4, 90.1, 82.7, 120.3, 95, 106.5, 95.9])
Soviet_revised_est=np.array([73.3, 71.7, 83.5, 69.5, 69.8, 68.4, 67.6, 75.0, 55.8, 97.4, 73.6, 73.2, 86.9])
Western_low_est=np.array([63., 62., 65., 56., 56., 65., 68., 75., 56., 97., 74., 73., 87.])
diff=Western_low_est-Soviet_initial_est
xticks=np.arange(np.min(Years_All),np.max(Years_All)+1,3)
ntries=100
#Number of points per day
Points_Per_Day=1
num_years= np.max(Years_All)-np.min(Years_All)
npts=(np.max(Years_All)-np.min(Years_All))*365*Points_Per_Day
Points_Skip = (np.min(Years)-np.min(Years_All))*365*Points_Per_Day
Points_To_Record = Points_Skip+(Years-np.min(Years))*365*Points_Per_Day-1
tspan = np.linspace(0,(np.max(Years_All)-np.min(Years_All))*12,npts) #Time in months
t2=tspan/12+np.min(Years_All)

tspan_months = np.linspace(0,np.max(Years_All)-np.min(Years_All),num_years*12)*12 #Time points are every month
v_all = np.zeros([npts,ntries])
e_all = np.zeros([npts,ntries])

t_exp=(Years-np.min(Years_All))*12 #Experimental data in points



v00=10
e0=5
mu0 = 1
sigma0 = 1
tshift0=50

pinit= np.array([tshift0,mu0])



#res = optimize.minimize(total_error, x0, args=params,tol=1e-8,method='COBYLA')
popt, pcov = curve_fit(v_asymptotic_params, t_exp, diff, p0=pinit)
params = popt
t_asympt = np.linspace(np.min(t_exp),np.max(t_exp),num_years+1)
tshift = params[0]
mu = params[1]
exp_fit = v_asymptotic_params(t_exp,tshift,mu)

#Orhnstein-Uhlenbeck process approximates well deviation from the asymptotic solution
#sigma^2/(2 \alpha)
var_experiments = np.var(diff- exp_fit)
#Above value is computed for data points in years
sigma = np.sqrt(2*alpha*var_experiments)/np.sqrt(12)
#Divide by sqrt(12) to go from 1/sqrt(years) to 1/sqrt(months)
nsims = 3

num_pts_sim = int(num_years*365.25)
t_end = num_years*12 #Computation time in months
e=np.zeros((num_pts_sim,nsims))
v=np.zeros((num_pts_sim,nsims))
B = np.diag([sigma, 0])

for j in range (0,nsims):
    #start at the asymptotic solution at t=0 (beginning of Stalin's reign)
    v0 = v_asymptotic_params(0,tshift,mu)[0]
    e0 = v_asymptotic_params(0,tshift,mu)[0]
    x0 = np.array([v0, e0])
    tspan = np.linspace(0, t_end, num_pts_sim)

    result = sdeint.itoint(f_SDE, sigma_SDE, x0, tspan)
    v[:,j] = result[:, 0]
    e[:,j] = result[:, 1]

tspan_years = tspan/12 #span in years

plt.figure(2)
plt.clf()
fig,ax = plt.subplots()
line_v  = ax.plot(tspan_years+np.min(Years_All),v[:,0:nsims],'r-',alpha=1,label='v',linewidth=1)
line_e  = ax.plot(tspan_years+np.min(Years_All),e[:,0:nsims],'b-',alpha=1,label='e',linewidth=1)
line_historic = ax.plot(Years, diff,'gx-',label="Historical data")
line_fit = ax.plot(Years,exp_fit,'ko-',label='Asymptotic solution fit')
#ax.legend([line_v, line_e, line_historic, line_fit],["True discrepancy (v)","Advisor informaiton (e)","Historic data", "Asymptotic fit"])
#ax.legend([line_v, line_e])
e_line = mlines.Line2D([], [], color='blue', label='e')
v_line = mlines.Line2D([], [], color='red', label='v')
legends = ax.get_legend_handles_labels()
#ax.legend(handler_map={e_line: HandlerLine2D(numpoints=4)})
ax.legend([line_e[0], line_v[0], line_historic[0],line_fit[0]],['e (3 realizations)','v (3 realizations)','Historic data','Asymptotic fit'])
# fig.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Difference in estimates')
ax.set_title('Results for grain production in USSR, 1928-1940')
#plt.legend(['Stochastic realization','Difference','Asymptotic solution fit','Historic data'])
plt.savefig('Wheat production variance.pdf')

