#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:05:41 2022

@author: vakhtang
"""
#This program requires sdeint installed:
#    https://pypi.org/project/sdeint/
import matplotlib.pyplot as plt
import numpy as np
import sdeint

sigma = 0.01
k = 0.05
alpha = 0.5
mu = 1
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
    G = k*phi(eta)**(gamma-1)/(phi(eta)**gamma+mu*np.abs(1-xi)**gamma)
    return G


B = np.diag([sigma, 0])


def f_SDE(x, t):
    v = x[0]
    e = x[1]
    g_val = k * np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt = -alpha*(v-e)
    dedt = g_val
    return np.array([dvdt, dedt])


def sigma_SDE(x, t):
    return B

#Initial conditions for simulations
v0 = 0.1
e0 = 0
x0 = np.array([v0, e0])
num_years=40
tend = num_years*12  # time in months

num_pts_sim = int(num_years*365.25)
tspan = np.linspace(0, tend, num_pts_sim)
result = sdeint.itoint(f_SDE, sigma_SDE, x0, tspan)
v = result[:, 0]
e = result[:, 1]
t = tspan
# Plot phase portrait of differential equation in the variables
# xi =(e-v)/v, eta=1/v


def dxideta(xi, eta):
    dxi = -G(xi, eta)*eta + alpha*xi**2-alpha*xi-eta**2*sigma**2
    deta = alpha*eta*xi+eta**3*sigma**2
    return [dxi, deta]


A = [[-alpha, alpha], [k, 0]]
lam = np.linalg.eig(A)[0]



kappa = np.max(lam)
plt.figure(1)
plt.clf()
plt.figure(constrained_layout=True)

#If a log-log plot desired, uncomment below
# plt.loglog(t, np.abs(v), 'b-', t, np.abs(e), 'r-',
#           t, 10*t**(1/gamma), 'k--', linewidth=1)
#v_arr=np.linspace(np.min(v),np.max(v),100)
np_asympt=40
va1=np.linspace(0.01,np.max(v),np_asympt)
va2=np.geomspace(0.01,np.max(v),np_asympt)
t_implicit_sol1=(np.log(va1)+mu/(gamma)*(va1**(gamma)))/k
t_implicit_sol2=(np.log(va2)+mu/(gamma)*(va2**(gamma)))/k

#Matching a constant in the solution: t(v_f)=t_f
vf=v[-1]
C=t[-1]-t_implicit_sol1[-1]
t_implicit_sol1+=C

C=t[-1]-t_implicit_sol2[-1]
t_implicit_sol2+=C

#If semilog plot is required to check for exponential growth, uncomment below
# plt.semilogy(t, np.abs(v), 'r-',t, np.abs(e), 'b-',
#               t, 0.15*np.exp(kappa*t), 'k--',
#               t_implicit_sol,v,'g-',linewidth=1)
plt.plot(t, v, 'r-',t, e, 'b-')
plt.scatter(t_implicit_sol1,va1,color='k',s=20,alpha=1)
plt.ylabel('v(t) and e(t)',fontsize=12)
plt.xlabel('Time, months',fontsize=12)
plt.legend(['v(t)', 'e(t)','Asymptotic'],fontsize=12)
plt.ylim([np.min(v)-0.1,v[-1]+0.1])
plt.title(r'$v(t)$ and $e(t)$ compared with the asymptotic solution',fontsize=12)
plt.xlim([0,np.max(t)])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig("Error_Variation_gamma=" + str(gamma)+"_linplot.pdf")


# Asymptoric solution for v
psi = k*np.abs(va2)/((1+mu*np.abs(va2)**gamma))
dpsidv = np.sign(va2)*k*(1+(1-gamma)*mu*np.abs(va2)**gamma) / \
    ((1+mu*np.abs(va2)**gamma)**2)
dpsidt = dpsidv*psi
psi1 = -psi/alpha-dpsidt/alpha**2
plt.figure(2)
ind = np.arange(int(num_pts_sim/5), num_pts_sim)


plt.figure(2)
plt.clf()
plt.figure(constrained_layout=True)

p = 1/gamma - 1

#If log-log solution is desired to check for power law growth, uncomment below
if gamma >= 1:
    plt.plot(t, v-e, 'r-', t_implicit_sol2, psi1, 'k-')
    #plt.loglog(t,np.abs(e-v),'r-', t,t**p,'k--',t,phi,'g-')
else:
    plt.plot(t, v-e, 'r-', t_implicit_sol2, psi1, 'k-')

    #plt.loglog(t,np.abs(e-v),'r-', t,t**p,'k--',t,phi,'g-')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title(r'Dictators view of the system',fontsize=12)
plt.legend(['v-e', 'Asymptotic solution'],fontsize=12)

plt.ylabel('v(t)-e(t)',fontsize=12)
plt.xlabel('Time, months',fontsize=12)
plt.xlim([0,np.max(t)])

plt.savefig("Diff_Error_gamma="+str(gamma)+".pdf")



xi_dim = 2
xi_h = xi_dim/50
eta_dim = 15
eta_h = eta_dim/50
xival, etaval = np.meshgrid(np.arange(-xi_dim/2, xi_dim, xi_h),
                            np.arange(0, eta_dim, eta_h))
[xidot, etadot] = dxideta(xival, etaval)

plt.figure(3)
plt.clf()
plt.figure(constrained_layout=True)

plt.streamplot(xival, etaval, xidot, etadot, density=[1, 0.5])
plt.plot([-xi_dim, xi_dim], [0, 0], 'k-', linewidth=2)
plt.plot([0, 1], [0, 0], 'bo', markersize=10)
xi_sol = (v-e)/v
eta_sol = 1/v
i_skip=1000
plt.plot(xi_sol[i_skip:], eta_sol[i_skip:], 'r-')
#Quiver is an alternative variant of plotting the phase portrait;
#we do not use it since streamplot looks nicer
#plt.plot(xi_sol[0], eta_sol[0], 'ro', markersize=10)
#plt.quiver(xival, etaval, 0.5*xidot, 0.5*etadot)
plt.xlim([-xi_dim/2, xi_dim])
plt.ylim([0, eta_dim])
plt.xlabel(r"$\xi$",fontsize=12)
plt.ylabel(r"$\eta$",fontsize=12)
plt.grid()
plt.title('Phase portrait of transformed equation',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.savefig("Phase_portrait_solution.pdf")
