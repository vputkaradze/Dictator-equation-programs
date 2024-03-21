

#import pandas as pd
#from os import listdir
#from os.path import isfile, join
#from datetime import datetime
#import csv
import matplotlib.pyplot as plt
import numpy as np
#import math
#import time
#from scipy.optimize import minimize
#import scipy.stats as st
from scipy.optimize import curve_fit
from scipy.special import lambertw
from scipy import stats
from scipy.optimize import fsolve

import sdeint

# A program simulating SDEs with several advisors

sigma = 0.2
alpha = 0.5
mu = 0.1
gamma = 1
num_advisors = 10
#Take decay_coeff = 1, 2, 3 with common_coeff =0.01 for nice pictures.
interaction_distance = 1

#Take common_coeff = 0.01 for pictures with interaction
#common_coeff =0 #for no interaction
common_coeff = 0

k = 0.05*(1+np.random.uniform(low=-0.1, high=0.1, size=num_advisors))
a =  -1
# a is the coefficient  e*dg/de = a*g for certain types of g with e in the denominator
eps = 0.1 #Sharpness of tanh


def beta_func(beta):
    lhs = (k + a*k*beta)
    rhs = (np.mean(k) + a*np.mean(k*beta))*(1+beta)
    return lhs-rhs


def A_func(A):
    lhs = A**2*k
    rhs = (np.mean(k*A))
    return lhs-rhs

def xi_func(x):
    f = 0.5*(np.tanh((x+interaction_distance)/eps)-np.tanh((x-interaction_distance)/eps))
    return f


A0 = np.ones(len(k))
A_sol = fsolve(A_func, A0)

beta0 = (1-A_sol)/A_sol
beta_sol = fsolve(beta_func, beta0)

#asymptotic_difference = - beta_sol
asymptotic_difference =   -(1-A_sol)/A_sol


def phi(x):
    v = np.abs(x)
    return v


def dW(delta_t: float) -> float:
    """
    Sample a random number at each call.
    """
    return np.random.default_rng().normal(loc=0.0, scale=np.sqrt(delta_t))


def G(xi, eta):
    numerator = np.abs(eta)**(gamma-1)  # numerator multiplied by |eta|^gamma
    denominator = np.abs(eta)**gamma + mu*np.abs(1-xi)**gamma
    G = numerator/denominator
    # G = |eta|**(gamma-1)/(|eta|**gamma+mu*np.abs(1-xi)**gamma)
    return G


B = np.diag(np.hstack([sigma, np.zeros(num_advisors)]))


def f_SDE(x, t):
    v = x[0]
    e = x[1:num_advisors+1]
    e1 = e*np.ones((len(e), 1))
    diff = e1 - e1.T
    screening_matr = xi_func(diff)*diff
    # Function g is negative to address Referee's concerns
    # Note that k included here rather than in the later expression
    g_val = - k * np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt = -alpha*(v-np.mean(e))
    neighbors = np.sum(xi_func(diff),axis=0)
    dedt = g_val - common_coeff*time_step/neighbors*np.sum(screening_matr,axis=0)
    dy = np.hstack([dvdt, dedt])
    return dy


def sigma_SDE(x, t):
    return B


def dxideta(xi, eta):
    dxi = k*G(xi, eta)*eta - (1-xi)*(alpha*xi+eta**2*sigma**2)
    deta = eta*(alpha*xi+eta**2*sigma**2)
    return [dxi, deta]


def v_asymptotic(tvalue, tshift):
    x = mu*np.exp(gamma*k*(tvalue-tshift*np.ones(np.size(tvalue))))
    # Solving equation z e^z = x
    z = lambertw(x)
    # we only have information about |v|, choosing minus
    v = -(z/mu)**(1/gamma)
    return np.real_if_close(v)


# Number of months should be about 100 for short-term graphs
# Using longer simulation times for variance plot
num_month_short_sim = 480
num_month_long_sim = 1000
time_step = 1./30 #Time step is needed for interaction term
num_pts_short_sim = num_month_short_sim*30  # time step is 1 day
num_pts_long_sim = num_month_short_sim*30  # time step is 1 day
nsims = 1
nshow = 1
# v0_arr=np.array([-5,-2.5,2.5,5])
v0_arr = np.array([5])
ncurves = len(v0_arr)
v_sol = np.zeros((num_pts_short_sim, nsims, ncurves))
e_sol = np.zeros((num_advisors, num_pts_short_sim, nsims, ncurves))

for j in range(ncurves):
    for m in range(nsims):
        v0 = v0_arr[j]
        #e0 = np.random.uniform(low=-10, high=10, size=num_advisors)
        e0 = np.linspace(-10,10,num_advisors)
        y0 = np.hstack([v0, e0])
        tspan_short = np.linspace(0, num_month_short_sim, num_pts_short_sim)
        result = sdeint.itoint(f_SDE, sigma_SDE, y0, tspan_short)
        v_sol[:, m, j] = result[:, 0]
        for s in range(num_advisors):
            e_sol[s, :, m, j] = result[:, s+1]

plt.figure(1)
plt.clf()
fig, ax = plt.subplots()
ind_plot = range(0, int(2*len(tspan_short)/3))


for j in range(ncurves):
    for s in range(num_advisors):
        line_e = ax.plot(tspan_short[ind_plot], e_sol[s, ind_plot,
                         0:nshow, j], 'g-', alpha=1, label='e', linewidth=1)

#plot the average of e
for j in range(ncurves):
    line_e_av =  ax.plot(tspan_short[ind_plot], np.mean(e_sol[:,ind_plot,
                         0:nshow, j],axis =1), 'b-', alpha=1, label='v', linewidth=2)


for j in range(ncurves):
    line_v = ax.plot(tspan_short[ind_plot], v_sol[ind_plot,
                     0:nshow, j], 'r-', alpha=1, label='v', linewidth=1)

ax.scatter(0*e0, e0, color='green', s=20)
ax.scatter(0*v0, v0, color='red', s=20)
ax.legend([line_e[0], line_e_av[0], line_v[0]], ['e', r'$\overline{e}$', 'v'])
ax.plot(tspan_short[ind_plot], 0*tspan_short[ind_plot], 'k--', linewidth=2)
ax.set_xlabel('Months')
ax.set_ylabel('e(t) and v(t)')
if (common_coeff>0):
    title_str = 'Simulation results: interaction with d='+str(interaction_distance)
else:
    title_str = 'Simulation results: no interaction'
ax.set_title(title_str)

# ax.set_ylim(np.min(v0_arr)-0.1,np.max(v0_arr)+0.1)
plt.savefig('Multi_advisor_Short term simulations_'+'c='\
            +str(common_coeff)+'_distance='+str(interaction_distance)+'.pdf')

# long simulations

v_sol2 = np.zeros((num_pts_long_sim, nsims, ncurves))
e_sol2 = np.zeros((num_advisors, num_pts_long_sim, nsims, ncurves))

for j in range(ncurves):
    for m in range(nsims):
        v0 = v0_arr[j]
        e0 = - np.random.uniform(low=-5, high=5, size=num_advisors)
        tspan_long = np.linspace(0, num_month_long_sim, num_pts_long_sim)
        y0 = np.hstack([v0, e0])
        result = sdeint.itoint(f_SDE, sigma_SDE, y0, tspan_long)
        v_sol2[:, m, j] = result[:, 0]
        for s in range(num_advisors):
            e_sol2[s, :, m, j] = result[:, s+1]

plt.figure(2)
plt.clf()
fig, ax = plt.subplots()

# Plot stochastic lines first since they are most messy

for j in range(ncurves):
    for s in range(num_advisors):
        line_e = ax.plot(
            tspan_long, e_sol2[s, :, 0:nshow, j], 'b-', alpha=0.5, label='e', linewidth=1)

for j in range(ncurves):
    line_v = ax.plot(
        tspan_long, v_sol2[:, 0:nshow, j], 'r-', alpha=1, label='v', linewidth=2)

ax.scatter(0*v0_arr, v0_arr, color='red', s=30)
ax.legend([line_e[0], line_v[0]], ['e', 'v'])
ax.plot(tspan_long, 0*tspan_long, 'k--', linewidth=1)
ax.set_xlabel('Months')
ax.set_ylabel('e(t) and v(t)')
ax.set_title('Simulation results, multiple advisors')
# ax.set_ylim(np.min(v0_arr)-0.1,np.max(v0_arr)+0.1)
plt.savefig('Multi_advisor_Long_term_simulations_realizations_long_e.pdf')


plt.figure(3)
plt.clf()
fig, ax = plt.subplots()
nt = len(tspan_long)
ind_plot = range(int(3*nt/4), nt)

for j in range(ncurves):
    #line_v_av =ax.plot(tspan_short,np.mean(v_sol[:,:,j],axis=1),'r-',alpha=1,label='E(v)',linewidth=1,markersize=5)
    denominator = 0+np.abs(np.mean(e_sol[:, :, :, j], axis=0))
    for s in range(num_advisors):

        diff_k = (e_sol2[s, :, :, j] -
                  np.mean(e_sol2[:, :, :, j], axis=0))/denominator
        line_e_av = ax.plot(tspan_long, diff_k, 'g-', alpha=1,
                            label=r'$(e-\overline{e})/|\overline{e}|$', linewidth=1, markersize=5)
        line_alpha = ax.plot(tspan_long[ind_plot], asymptotic_difference[s]*np.ones(
            len(ind_plot)), 'r-', alpha=1, label='asymptotic', linewidth=1, markersize=5)

ax.plot(tspan_long, 0*tspan_long, 'k--', linewidth=1)

# ax.scatter(0*v0_arr,v0_arr,color='red',s=30)
ax.legend([line_e_av[0], line_alpha[0]], [
          r'$\frac{\Delta e_k}{|\overline{e}|}$', 'asymptotic'])
ax.set_xlabel('Months')
ax.set_ylabel(r'$\frac{\Delta e_k}{10+|\overline{e}_k|}$')
plt.ylim(-0.25,0.25)
if (common_coeff>0):
    title_str = 'Simulation results: interaction with d='+str(interaction_distance)
else:
    title_str = 'Simulation results: no interaction'
ax.set_title(title_str)
plt.savefig('Multi_advisor_Short term simulations difference.pdf',
            bbox_inches='tight')

plt.figure(4)
plt.clf()
fig, ax = plt.subplots()

for j in range(ncurves):
    line_v_av = ax.plot(tspan_short, np.mean(
        v_sol[:, :, j], axis=1), 'r-', alpha=1, label='E(v)', linewidth=1, markersize=5)
    for s in range(num_advisors):
        line_e_av = ax.plot(tspan_short, np.mean(
            e_sol[s, :, :, j], axis=1), 'b-', alpha=1, label='E(e)', linewidth=1, markersize=5)

ax.plot(tspan_short, 0*tspan_short, 'k--', linewidth=1)
ax.scatter(0*v0_arr, v0_arr, color='red', s=30)
ax.legend([line_e_av[0], line_v_av[0]], ['E[e]', 'E[v]'])
ax.set_xlabel('Months')
ax.set_ylabel('E[e](t) and E[v](t)')
ax.set_title('Simulation results, short-term: averaged')
plt.savefig('Multi_advisor_Short_term_simulations_mean.pdf')


#fitting slopes to slopes for e_k
slopes = np.zeros(num_advisors)
for s in range(num_advisors):
    p = np.polyfit(tspan_long[ind_plot], e_sol2[s, ind_plot, 0, 0], 1)
    slopes[s]=p[0]

rel_diff_s = (slopes - np.mean(slopes))/np.mean(slopes)

plt.figure(5)
plt.clf()
line_asympt = plt.scatter(np.arange(num_advisors)+1,beta_sol,marker='o',color = 'r',s=20)
line_measured = plt.scatter(np.arange(num_advisors)+1,rel_diff_s,marker='x',color = 'b',s=50)

plt.legend([line_asympt, line_measured], ['Asymptotic', 'Measured'])

plt.xticks(np.arange(1, 10, step=1))
plt.title(r'Limit of $\Delta e_k/\overline{e}$ as $t \rightarrow \infty$')
plt.ylabel(r'Limit $\Delta e_k/\overline{e}$')
plt.xlabel('Advisor index')
if common_coeff==0:

    plt.savefig('Miltiple_advisors_asymptotic_vs_measured.pdf',bbox_inches='tight')
else:
    #in case the first and second advisor belong to different groups, plot two lines
    plt.plot(np.arange(1,11),rel_diff_s[0]*np.ones(10),'k:')
    plt.plot(np.arange(1,11),rel_diff_s[1]*np.ones(10),'k:')

    plt.savefig('Miltiple_advisors_asymptotic_vs_measured_c='+str(common_coeff)+'.pdf',bbox_inches='tight')


