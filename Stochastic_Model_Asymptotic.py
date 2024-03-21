

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

import sdeint

sigma = 0.2
k = 0.05
alpha = 0.5
mu = 0.1
gamma = 1


def phi(x):
    v = np.abs(x)
    return v


def dW(delta_t: float) -> float:
    """
    Sample a random number at each call.
    """
    return np.random.default_rng().normal(loc=0.0, scale=np.sqrt(delta_t))


def G(xi, eta):
    numerator=np.abs(eta)**(gamma-1) #numerator multiplied by |eta|^gamma
    denominator = np.abs(eta)**gamma + mu*np.abs(1-xi)**gamma
    G=numerator/denominator
    #G = |eta|**(gamma-1)/(|eta|**gamma+mu*np.abs(1-xi)**gamma)
    return G


B = np.diag([sigma, 0])


def f_SDE(x, t):
    v = x[0]
    e = x[1]
    #Function g is negative to address Referee's concerns
    #Note that k included here rather than in the later expression
    g_val = - k * np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt = -alpha*(v-e)
    dedt = g_val
    return np.array([dvdt, dedt])


def sigma_SDE(x, t):
    return B

def dxideta(xi, eta):
    dxi = k*G(xi, eta)*eta -  (1-xi)*(alpha*xi+eta**2*sigma**2)
    deta = eta*(alpha*xi+eta**2*sigma**2)
    return [dxi, deta]

def v_asymptotic(tvalue,tshift):
    x=mu*np.exp(gamma*kappa*(tvalue-tshift*np.ones(np.size(tvalue))))
    #Solving equation z e^z = x
    z=lambertw(x)
    #we only have information about |v|, choosing minus
    v=-(z/mu)**(1/gamma)
    return np.real_if_close(v)


#Number of months should be about 100 for short-term graphs
#Using longer simulation times for variance plot
num_month_short_sim = 150
num_month_long_sim = 150

num_pts_short_sim = num_month_short_sim*30 #time step is 1 day
num_pts_long_sim = num_month_short_sim*30 #time step is 1 day
nsims = 100
nshow = 3
v0_arr=np.array([-5,-2.5,2.5,5])
ncurves = len(v0_arr)
v_sol = np.zeros((num_pts_short_sim, nsims,ncurves))
e_sol = np.zeros((num_pts_short_sim, nsims,ncurves))

for j in range(ncurves):
    for m in range (nsims):
        v0=v0_arr[j]
        e0=0
        tspan_short = np.linspace(0, num_month_short_sim, num_pts_short_sim)
        result = sdeint.itoint(f_SDE, sigma_SDE, [v0,e0], tspan_short)
        v_sol[:,m,j] = result[:, 0]
        e_sol[:,m,j] = result[:, 1]

v_sol2 = np.zeros((num_pts_long_sim, nsims,ncurves))
e_sol2 = np.zeros((num_pts_long_sim, nsims,ncurves))

for j in range(ncurves):
    for m in range (nsims):
        v0=v0_arr[j]
        e0=-0.1*v0_arr[j]
        tspan_long = np.linspace(0, num_month_long_sim, num_pts_long_sim)
        result = sdeint.itoint(f_SDE, sigma_SDE, [v0,e0], tspan_long)
        v_sol2[:,m,j] = result[:, 0]
        e_sol2[:,m,j] = result[:, 1]

ic_colors_v = ['r','g','m','y']

plt.figure(9)
plt.clf()
fig,ax = plt.subplots()
line_label = []
labels = []
#Plot stochastic lines first since they are most messy
for j in range(ncurves):
    col_val = ic_colors_v[j]+'-'
    line_v = ax.plot(tspan_short,v_sol[:,0:nshow,j],col_val,alpha=1,label='v',linewidth=1)
    line_label.append(line_v[0])
    ax.scatter(0*v0_arr[j],v0_arr[j],color=ic_colors_v[j] ,s=30)
    labels.append('v(t) realizations: v(0)='+str(v0_arr[j]))
for j in range(ncurves):
    line_e  = ax.plot(tspan_short,e_sol[:,0:nshow,j],'b-',alpha=1,label='e',linewidth=1)

line_label.append(line_e[0])
labels.append('corresponding e(t)')

ax.legend(line_label,labels)
ax.plot(tspan_short,0*tspan_short,'k--',linewidth=1)
ax.set_xlabel('Months')
ax.set_ylabel('e(t) and v(t)')
ax.set_xlim([-5,140])
ax.set_title('Simulation results, short-term: realizations: e(0)=0')
ax.set_ylim(np.min(v0_arr)-10,np.max(v0_arr)+1)
plt.savefig('Short_term_simulations_realizations.pdf')


plt.figure(13)
plt.clf()
fig,ax = plt.subplots()
line_label = []
labels = []

#Plot stochastic lines first since they are most messy
for j in range(ncurves):
    col_val = ic_colors_v[j]+'-'
    line_v  = ax.plot(tspan_long,v_sol2[:,0:nshow,j],col_val,alpha=1,label='v',linewidth=1)
    line_label.append(line_v[0])
    ax.scatter(0*v0_arr[j],v0_arr[j],color=ic_colors_v[j] ,s=30)
    labels.append('v(t) realizations: v(0)='+str(v0_arr[j]))
    ax.scatter(0*v0_arr[j],v0_arr[j],color= ic_colors_v[j] ,s=30)


for j in range(ncurves):
    line_e  = ax.plot(tspan_long,e_sol2[:,0:nshow,j],'b-',alpha=1,label='e',linewidth=1)

line_label.append(line_e[0])
labels.append('corresponding e(t)')

ax.legend(line_label,labels)

ax.plot(tspan_long,0*tspan_long,'k--',linewidth=1)
ax.set_xlabel('Months')
ax.set_ylabel('e(t) and v(t)')
ax.set_title('Simulation results, short-term: realizations e(0)=-0.1v(0)')
ax.set_ylim(np.min(v0_arr)-14,np.max(v0_arr)+1)
ax.set_xlim(-2,145)
plt.savefig('Long_term_simulations_realizations_alternative_e.pdf')



plt.figure(10)
plt.clf()
fig,ax = plt.subplots()

for j in range(ncurves):
    line_e_av =ax.plot(tspan_short,np.mean(e_sol[:,:,j],axis=1),'b-',alpha=1,label='E(e)',linewidth=1,markersize=5)
    line_v_av =ax.plot(tspan_short,np.mean(v_sol[:,:,j],axis=1),'r-',alpha=1,label='E(e)',linewidth=1,markersize=5)

ax.plot(tspan_short,0*tspan_short,'k--',linewidth=1)
ax.scatter(0*v0_arr,v0_arr,color='red',s=30)
ax.legend([line_e_av[0], line_v_av[0]],['E[e] (100 realizations)','E[v] (100 realizations)'])
ax.set_xlabel('Months')
ax.set_ylabel('E[e](t) and E[v](t)')
ax.set_title('Simulation results, short-term: averaged')
plt.savefig('Short term simulations mean.pdf')



#plt.legend(['Stochastic realization','Difference','Asymptotic solution fit','Historic data'])


num_years=40
tend = num_years*12  # time in months

num_pts_sim = int(num_years*365.25)

e=np.zeros((num_pts_sim,nsims))
v=np.zeros((num_pts_sim,nsims))
for j in range (0,nsims):
    v0 = 0
    e0 = 0
    x0 = np.array([v0, e0])
    tspan = np.linspace(0, tend, num_pts_sim)
    result = sdeint.itoint(f_SDE, sigma_SDE, x0, tspan)
    v[:,j] = result[:, 0]
    e[:,j] = result[:, 1]

t = tspan
# Plot phase portrait of differential equation in the variables
# xi =(e-v)/v, eta=1/v


A = [[-alpha, alpha], [k, 0]]
lam = np.linalg.eig(A)[0]
kappa = np.max(lam)
# t = np.linspace(0, num_pts_sim, num_pts_sim)*h
C1 = np.zeros(nsims)
C2 = np.zeros(nsims)

# plt.loglog(t, np.abs(v), 'b-', t, np.abs(e), 'r-',
#           t, 10*t**(1/gamma), 'k--', linewidth=1)
#v_arr=np.linspace(np.min(v),np.max(v),100)
np_asympt=40
t_asympt = np.linspace(0,np.max(t),np_asympt)
tv_shift = np.zeros(nsims)
te_shift = np.zeros(nsims)
#We are fitting only the last half of the data because the fit is asymptotic
ind_fit = np.arange(int( 2*np.size(t)/3 ),np.size(t))
for j in range(0,nsims):
    #Matching a constant in the solution: tshift using best fit
    #tv are fits to v; te are fits to e data. They should be close, but not the same
    popt, pcov = curve_fit(v_asymptotic, t[ind_fit], v[ind_fit,j])
    tv_shift[j]=popt[0]
    popt, pcov = curve_fit(v_asymptotic, t[ind_fit], e[ind_fit,j])
    te_shift[j]=popt[0]


# plt.semilogy(t, np.abs(v), 'r-',t, np.abs(e), 'b-',
#               t, 0.15*np.exp(kappa*t), 'k--',
#               t_implicit_sol,v,'g-',linewidth=1)
plt.figure(1)
plt.clf()
plt.figure(constrained_layout=True)

nplots=3
indplot=[1,np.argmin(te_shift), np.argmax(te_shift)]
for j in range (0,nplots):
    plt.plot(t, v[:,indplot[j]], 'r-',t, e[:,indplot[j]], 'b-',linewidth=1,alpha=0.5)
    #plt.plot(t_implicit_sol,va,'k-',linewidth=1)
    plt.scatter(t_asympt,v_asymptotic(t_asympt,te_shift[indplot[j]]),color='k',s=30,alpha=1)
#plt.plot(t, v, 'r-', t, e, 'b--', linewidth=1)
plt.ylabel('v(t) and e(t)',fontsize=12)
plt.xlabel('Time, months',fontsize=12)
plt.legend(['v(t)', 'e(t)','Asymptotic'],fontsize=12)
plt.ylim([np.min(v)-0.1,np.max(v)+0.1])
plt.title(r'$v(t)$ and $e(t)$ compared with the asymptotic solution',fontsize=12)
plt.xlim([0,np.max(t)])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig("Error_Variation_gamma=" + str(gamma)+"_linplot.pdf")

plt.figure(12)
plt.clf()
fig,ax = plt.subplots()
t0=np.linspace(np.min(te_shift),np.max(te_shift),10)
ax.set_aspect('equal')
ax.plot(te_shift,tv_shift,'ro',t0,t0,'k-')
ax.legend(['Numerical results','Equal'])
ax.set_ylabel(r'Fitted shift in $v(t)$')
ax.set_xlabel(r'Fitted shift in $e(t)$')
plt.savefig('tv_vs_te.pdf')



#Computing variance from asymptotics
plt.figure(11)
plt.clf()
fig,ax = plt.subplots()
e_diff = 0*e
v_diff = 0*v
for j in range (0,nsims):
    #In principle, one can take te_shift and tv_shift to be the same
    e_diff[:,j]=e[:,j] - v_asymptotic(t,te_shift[j])
    v_diff[:,j]=v[:,j] - v_asymptotic(t,te_shift[j])

ax.plot(t,np.var(e_diff,axis=1),'b-',t,np.var(v_diff,axis=1),'r-')
expected_var = sigma**2/(2*alpha)
ax.plot(t,np.ones(np.size(t))*expected_var,'k-')

# for j in range(ncurves):
#     line_e_av =ax.plot(tspan,np.var(e[:,:,j],axis=1),'b-',alpha=1,label='E(e)',linewidth=1,markersize=5)
#     line_v_av =ax.plot(tspan,np.var(v[:,:,j],axis=1),'r-',alpha=1,label='E(e)',linewidth=1,markersize=5)

# ax.plot(tspan,0*tspan,'k--',linewidth=1)
ax.legend([r'Var[$e-e_{asympt}$] (100 realizations)',r'Var[$v-v_{asympt}$] (100 realizations)',r'Expected variance $\sigma^2/(2 \alpha)$'])
ax.set_xlabel('Months')
ax.set_ylabel('Var[e](t) and Var[v](t)')
ax.set_title('Variance of v and e')
plt.savefig('Short term simulations variance.pdf')



psi1 = np.zeros((np_asympt,nsims)) #Initializing the size of the array
for j in range(0,nsims):
    v_as_val = v_asymptotic(t_asympt, te_shift[j])
    psi = -k*np.abs(v_as_val)/((1+mu*np.abs(v_as_val)**gamma))
    dpsidv = - np.sign(v_as_val)*k*(1+(1-gamma)*mu*np.abs(v_as_val)**gamma) / \
    ((1+mu*np.abs(v_as_val)**gamma)**2)
    dpsidt = dpsidv*psi
    psi1[:,j] = -psi/alpha-dpsidt/alpha**2

# zeta = (e-v)/psi1

# plt.plot(t_implicit_sol, zeta[ind]-1, 'k-')

plt.figure(2)
plt.clf()
plt.figure(constrained_layout=True)

p = 1/gamma - 1

for j in range (0,1):
    #log-log plots may be chosen differently if desired
    #Linear plots are the same
    if gamma >= 1:
        plt.plot(t, v[:,j]-e[:,j], 'r-',linewidth=1,zorder=1)
        plt.plot(t,np.mean(v-e,1),'b-',linewidth=1,zorder=2)
        #plt.scatter(t_asympt,psi1[:,j],color='k',s=30,alpha=1)
        #plt.loglog(t,np.abs(e-v),'r-', t,t**p,'k--',t,phi,'g-')
    else:
        plt.plot(t, v[:,j]-e[:,j], 'r-', t_asympt, psi1[:,j], 'k-')
        #plt.loglog(t,np.abs(e-v),'r-', t,t**p,'k--',t,phi,'g-')
plt.scatter(t_asympt,psi1[:,j],color='k',s=30,alpha=1,zorder=3)

plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title(r'Dictator''s view of the system',fontsize=12)
plt.legend(['v-e', 'E[v-e]','Asymptotic solution'],fontsize=12)

#plt.semilogy(t,np.abs(e),'r-',t, np.exp(kappa*t),'k--',linewidth=1)
# plt.plot(t,e,'r-',linewidth=1)

plt.ylabel('v(t)-e(t)',fontsize=12)
plt.xlabel('Time, months',fontsize=12)
plt.xlim([0,np.max(t)])

plt.savefig("Diff_Error_gamma="+str(gamma)+".pdf")



xi_dim = 2
xi_h = xi_dim/50
eta_dim = 2
eta_h = eta_dim/100
xival, etaval = np.meshgrid(np.arange(-xi_dim/2, xi_dim, xi_h),
                            np.arange(- eta_dim, 0, eta_h))
[xidot, etadot] = dxideta(xival, etaval)

plt.figure(3)
plt.clf()
plt.figure(constrained_layout=True)

plt.streamplot(xival, etaval, xidot, etadot, density=[1, 1])
plt.plot([-xi_dim, xi_dim], [0, 0], 'k-', linewidth=2)
plt.plot([0, 1], [0, 0], 'bo', markersize=10)
ind_plot=1 #could be any number; some of the trajectories are very messy in transformed coordinates
xi_sol = (v[:,ind_plot]-e[:,ind_plot])/v[:,ind_plot]
eta_sol = 1/v[:,ind_plot]
i_skip=1700 #Only plotting the last part on the curve - initial part looks very messy in transformed coordinates
plt.plot(xi_sol[i_skip:], eta_sol[i_skip:], 'r-')
#plt.plot(xi_sol[0], eta_sol[0], 'ro', markersize=10)
#plt.quiver(xival, etaval, 0.5*xidot, 0.5*etadot)
plt.xlim([-xi_dim/2, xi_dim])
plt.ylim([- eta_dim, 0])
plt.xlabel(r"$\xi$",fontsize=12)
plt.ylabel(r"$\eta$",fontsize=12)
plt.grid()
plt.title('Phase portrait of transformed equation',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig("Phase_portrait_solution.pdf")


