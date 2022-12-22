#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:05:41 2022

@author: vakhtang
"""
#This program requires sdeint installed:
#    https://pypi.org/project/sdeint/
#A program to compute the first intersection of v=0 starting with different initial conditions
import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy import integrate

def f_SDE(x,t):
    v=x[0]
    e=x[1]
    g_val = k* np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt=-alpha*(v-e)
    dedt=g_val
    return np.array([dvdt,dedt])

def sigma_SDE(x,t):
    return B

#numtests is the number of points to run. Note that the program takes quite a lont time to run!
numtests=10
#alpha and k are fixed to the values in the paper
# Time is measured in units of 1/alpha, so t=1 is equivalent to 2 month
alpha = 0.5 #1/alpha is response time in months
k = 0.05
#All alpha and k are taken to be the same for all tests
alpha_arr = alpha*np.ones(numtests)
k_arr     = k*np.ones(numtests)

to_plot_histogram=False #Make this True to plot the histogram.
#If that value is true, the first histogram is plotted with fixed values
sigma_arr = np.random.uniform(0.01,0.1,numtests)
mu_arr    = np.random.uniform(0.1,2,numtests)
gamma_arr = np.random.uniform(0.5,1.5,numtests)
v0_arr    = np.random.uniform(-0.2,-0.01,numtests)

#Calculation of eigenvalues for the approximate value in the paper
A=np.array([[1,-1],[k,0]])
lam,v = np.linalg.eig(A)
vi=np.linalg.inv(v)
sigma_0=np.matmul(vi,np.array([[1],[0]]))
max_eig_index=np.argmax(lam)

t_end=20.0 #Number in months
#Increase this to 1000 to get better results: SLOW!
num_tries=100
num_pts_sim = int(t_end*30) #Delta t is 1 day

t_cross=np.zeros((numtests,num_tries))
t_predict=np.zeros(numtests)

#Analytic formula used in the paper
t_times_p = lambda t , x, lam_val: t*np.abs(x)/np.sqrt(2*np.pi)*(lam_val/np.sinh(lam_val*t))**(3./2) \
                        *np.exp(- lam_val*x**2*np.exp(-lam_val*t)/(2*np.sinh(lam_val*t)) \
                                + lam_val*t/2)



for m in range (0,numtests):

    sigma_transform=sigma_0[max_eig_index]*sigma_arr[m] #The real value of sigma



    if ((to_plot_histogram) & (m==0)):
        #Plot histogram of one realization
        sigma=0.01
        k=0.05
        alpha=0.5
        mu=1
        gamma=1
        v0=-0.05
        lam_val = lam[max_eig_index]*alpha
        sigma_transform=sigma_0[max_eig_index]*sigma
    else:
        sigma = sigma_arr[m]
        k = k_arr[m]
        alpha = alpha_arr[m]
        mu = mu_arr[m]
        gamma = gamma_arr[m]
        v0=v0_arr[m]
        sigma_transform=sigma_0[max_eig_index]*sigma_arr[m] #The real value of sigma

    B=np.diag([sigma,0])
    e0=0
    x0=np.array([v0,e0])
    for j in range(0,num_tries):

        tspan=np.linspace(0,t_end,num_pts_sim)
        result = sdeint.itoint(f_SDE, sigma_SDE, x0, tspan)
        v=result[:,0]
        e=result[:,1]
        t=tspan
        #Find the first occurrence where v changes sign the first time
        ind_change = np.where(v[:-1] * v[1:] < 0 )[0][0]
        t_cross[m,j]=tspan[ind_change]

    print("Test group number ",m," Change of sign occurred at t=",np.mean(t_cross[m,:]))
    if (to_plot_histogram):
        lam_val=lam[max_eig_index]*alpha
    else:
        lam_val=lam[max_eig_index]*alpha_arr[m]

    xi0_vec=np.matmul(vi,np.array([[v0],[0]]))
    x=xi0_vec[max_eig_index]/sigma_transform

    #computing mean exit from the formula
    t_predict[m], err = integrate.quad(t_times_p, 0, 100, args=(x,lam_val))
    if ((m==0) and to_plot_histogram):
        #Plot histogram of solutions
        plt.figure(1)
        plt.clf()
        plt.figure(constrained_layout=True)
        tmax=10
        plt.hist(t_cross[0,:],bins=40,density=True,range=[0,tmax])
        #Plotting theoretical graph of probability transition from v0 to 0
        # Formula (2.8) from L. Alili, P. Patie and J.L. Pedersen



        t1=t[1:]
        p=np.abs(x)/np.sqrt(2*np.pi)*(lam_val/np.sinh(lam_val*t1))**(3./2) \
            *np.exp(\
                - lam_val*x**2*np.exp(-lam_val*t1)/(2*np.sinh(lam_val*t1)) \
                + lam_val*t1/2 \
            )
        #Done in order to avoid singularity at t=0
        p=np.insert(p,0,0) #inserting value of 0 in the beginning
        #plt.figure(1)
        plt.plot(t,p,'r-')
        plt.xlim([0,tmax])
        plt.title('Histogram of first time crossing',fontsize=12)
        plt.xlabel('First crossing time, months',fontsize=12)
        plt.ylabel('Probability density',fontsize=12)
        plt.legend(['Theory','Simulation'],fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.savefig('HistogramFirstCrossing.pdf')

mean_t_cross=np.mean(t_cross,axis=1)


valmax=np.max([np.max(mean_t_cross),np.max(t_predict)])+0.2
t1=np.linspace(0,valmax,10)
fig=plt.figure(2)
plt.clf()
fig, ax = plt.subplots(1, 1)
ax.plot(mean_t_cross,t_predict,'bo',t1,t1,'r-')
# x=np.linspace(0,valmax,10)
# plt.plot(t_predict1,mean_t_cross,'ro',x,x,'k-')

ax.set_title('Mean zero crossing time for v(0)<0',fontsize=12)
plt.xlabel('Measured',fontsize=12)
plt.ylabel('Predicted',fontsize=12)
ax.legend(['Experiment','Ideal'],fontsize=12)
ax.set(xlim=(0, valmax), ylim=(0, valmax))
ax.set_aspect('equal')
fig.tight_layout()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig('Predicted_vs_measured_zero_crossing.pdf')

