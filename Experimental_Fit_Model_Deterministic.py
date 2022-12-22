#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Fit both the model and the noise
"""
Created on Mon Jan 24 10:05:41 2022

@author: vakhtang
"""
#This program requires sdeint installed:
#    https://pypi.org/project/sdeint/
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import sdeint
from scipy import integrate


alpha0 = 0.5 #In months^{-1}
k0 = 0.05 #In months^{-1}

gamma0 = 1
v00=5
e00=5
mu0 = 0.5
sigma0 = 3

##Estimates from Wheatcroft and Davies, 1994, Wheat production in USSR, 1913-1940 (Mill Tons)
#Year; Soviet estimate reported; Soviet Estimate Revised; Low Western Estimate
Years=np.arange(1928,1941,1)
t0 =np.min(Years)
Years_All = np.arange(1922,1941,1) #Stalin comes to power at 1922
Soviet_initial_est=np.array([73.3, 71.7, 83.5, 69.5, 69.8, 89.8, 89.4, 90.1, 82.7, 120.3, 95, 106.5, 95.9])
Soviet_revised_est=np.array([73.3, 71.7, 83.5, 69.5, 69.8, 68.4, 67.6, 75.0, 55.8, 97.4, 73.6, 73.2, 86.9])
Western_low_est=np.array([63., 62., 65., 56., 56., 65., 68., 75., 56., 97., 74., 73., 87.])
diff=Soviet_initial_est-Western_low_est
xticks=np.arange(np.min(Years_All),np.max(Years_All)+1,3)
ntries=100
#Number of points per day
Points_Per_Day=1
num_years = np.max(Years_All)-np.min(Years_All)
npts=(np.max(Years_All)-np.min(Years_All))*365*Points_Per_Day
Points_Skip = (np.min(Years)-np.min(Years_All))*365*Points_Per_Day
Points_To_Record = Points_Skip+(Years-np.min(Years))*365*Points_Per_Day-1
tspan = np.linspace(0,(np.max(Years_All)-np.min(Years_All))*12,npts) #Time in months
tspan_months = np.linspace(0,np.max(Years_All)-np.min(Years_All),num_years*12)*12 #Time points are every month
v_all = np.zeros([npts,ntries])
e_all = np.zeros([npts,ntries])

t_exp=(Years-np.min(Years_All))*12. #Experimental data in points


def f_ODE(t,x,alpha,k,mu,gamma):
    v=x[0]
    e=x[1]
    g_val = k* np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt=-alpha*(v-e)
    dedt=g_val
    return np.array([dvdt,dedt])



def total_error(x,params):
    #A function for matching the error in the mean deviation and variance of the asymptotic function
    (v0,e0,mu)=x
    (alpha,k,sigma,gamma,to_plot)=params
    B=np.diag([sigma,0])
    def f_SDE(x,t):
        v=x[0]
        e=x[1]
        g_val = k* np.abs(v)/(1+mu*abs(e)**gamma)
        dvdt=-alpha*(v-e)
        dedt=g_val
        return np.array([dvdt,dedt])

    def sigma_SDE(x,t):
        return B

    y0=np.array([v0,e0])
    t_interval=(0,np.max(tspan_months)+1)
    sol=integrate.solve_ivp(f_ODE,t_interval,y0,t_eval=t_exp,args=(alpha,k,mu,gamma))
    v_deterministic=sol.y[0,:]

    #print("values of v found ",v_vals)
    L2error=np.sum((v_deterministic-diff)**2)
    #print("vals=",v_deterministic," diff=",diff)
    print("L2error=",L2error)
    #computing the mean and variance


    if (to_plot>0) :
        y0=np.array([v0,e0])
        t_interval=(0,np.max(tspan_months)+1)
        sol=integrate.solve_ivp(f_ODE,t_interval,y0,t_eval=tspan_months,args=(alpha,k,mu,gamma))

        v_asympt=sol.y[0,:]
        #print("Asymptotic values of v  ",v_asympt)
        t2=tspan/12+np.min(Years_All)
        #result = sdeint.itoEuler(f_SDE, sigma_SDE, y0, tspan-np.min(Years_All))
        result=sdeint.itoEuler(f_SDE,sigma_SDE,y0,tspan)
        v=result[:,0]
        e=result[:,1]

        plt.figure(1)
        plt.clf()
        plt.plot(tspan_months/12+np.min(Years_All),v_asympt,'g--',markersize=6,linewidth=2)
        plt.plot(t2,v,'r-',linewidth=0.5)
        plt.plot(t2,e,'b-',linewidth=2)
        plt.plot(Years,diff,'ko-',markersize=6, linewidth=2)
        plt.xlabel('Year',fontsize=12)
        plt.ylabel('Statistics Error, MT',fontsize=12)
        plt.title('Grain production: Soviet minus Western Estimate',fontsize=12)
        plt.legend(['Asymptotic: v','Simulations: v','Simulations: e','Historic Data'],fontsize=12,loc='upper left')
        plt.yticks(fontsize=12)
        plt.xticks(ticks=xticks,fontsize=12)

        plt.savefig('Grain_production.pdf')

    return L2error


#x0 are the parameters of the noiseless trajectory that are fitted, which are:
#initial conditions (v0,e0) and the value mu.
#The value of gamma is taken to be constant.
#The value of sigma is fitted to match the variance
x0=[v00,e00,mu0]
to_plot=0
params = np.array([alpha0,k0,sigma0,gamma0,to_plot])

res = optimize.minimize(total_error, x0, args=params,tol=1e-8,method='Nelder-Mead')
#If only one parameter is fitted, use the minimize scalar routine below
#res = optimize.minimize_scalar(total_error, bounds=[-400,100], args=params,method='Bounded')
xs=res.x
print("Solution=",res.x)
L2=total_error(xs,params)
sigma0=np.sqrt(L2/len(diff)*2*alpha0)
to_plot=1
print('sigma=',sigma0)
params = np.array([alpha0,k0,sigma0,gamma0,to_plot])
total_error(xs,params)

# params_sol[-1]=to_plot
# total_error(params_sol)
# (k,v0)=res.x
# K=k/alpha
# A=np.array([[1,-1],[K,0]])
# lam,v = np.linalg.eig(A)
# vi=np.linalg.inv(v)
# sigma_0=np.matmul(vi,np.array([[1],[0]]))
# max_eig_index=np.argmax(lam)
# lam_max=lam[max_eig_index]


# plt.figure(2)
# plt.clf()
# plt.plot(Years, var_arr,'bo-')
# plt.xlabel('Year')
# plt.ylabel('Variance')
# plt.title('Variance of wheat production')
# plt.legend(['Difference'])
# plt.savefig('Wheat production variance.pdf')

