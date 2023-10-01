
#A program to compute the first intersection of v=0
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import sdeint
from scipy import integrate

def f_SDE(x,t):
    v=x[0]
    e=x[1]
    g_val = -k* np.abs(v)/(1+mu*abs(e)**gamma)
    dvdt=-alpha*(v-e)
    dedt=g_val
    return np.array([dvdt,dedt])

def sigma_SDE(x,t):
    return B

v0_init = np.array([0.1, 1])


numtests=100
#Use numtests=1 if plot_histograms = True;
#Use numtests = the desired number of points (e.g., 100) if producing the mean zero crossing plots
to_plot_histogram=False #Make this True to plot the histogram

num_tries=100 #Number of tries for each histogram:
    #1000 if PlotHistogram = True; 100 if PlotHistogram = False

#alpha and k are fixed to the values in the paper
# Time is measured in units of 1/alpha, so t=1 is equivalent to 1 month
alpha = 0.5 #1/alpha is response time in months
k = 0.05
alpha_arr = alpha*np.ones(numtests)
k_arr     = k*np.ones(numtests)


# sigma_arr = 0.01*np.ones(numtests)
# mu_arr    = 0.1*np.ones(numtests)
# gamma_arr = 1*np.ones(numtests)
# v0_arr    = -0.1*np.ones(numtests)

sigma_arr = np.random.uniform(0.01,0.1,numtests)
mu_arr    = np.random.uniform(0.1,2,numtests)
gamma_arr = np.random.uniform(0.5,1.5,numtests)
v0_arr    = np.random.uniform(0.1,1,numtests)

A=np.array([[1,-1],[k,0]])
lam,v = np.linalg.eig(A)
vi=np.linalg.inv(v)
sigma_0=np.matmul(vi,np.array([[1],[0]]))
max_eig_index=np.argmax(lam)

t_end=100.0 #Number in months
num_pts_sim = int(t_end*30) #Delta t is 1 day

t_cross=np.zeros((numtests,num_tries))
t_predict=np.zeros(numtests)


t_times_p = lambda t , x, lam_val: t*np.abs(x)/np.sqrt(2*np.pi)*(lam_val/np.sinh(lam_val*t))**(3./2) \
                        *np.exp(- lam_val*x**2*np.exp(-lam_val*t)/(2*np.sinh(lam_val*t)) \
                                + lam_val*t/2)



for m in range (0,numtests):

    sigma_transform=sigma_0[max_eig_index]*sigma_arr[m] #The real value of sigma



    if (to_plot_histogram):
        sigma=0.1
        k=0.05
        alpha=0.5
        mu=1
        gamma=1
        v0=v0_init[1] #Use either 0 or 1 to produce results with different initial values
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
        plt.title('Histogram of first time crossing with v(0)='+str(v0),fontsize=12)
        plt.xlabel('First crossing time, months',fontsize=12)
        plt.ylabel('Probability density',fontsize=12)
        plt.legend(['Theory','Simulation'],fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.savefig('HistogramFirstCrossing_v0_'+str(v0)+'.pdf')

mean_t_cross=np.mean(t_cross,axis=1)


valmax=np.max([np.max(mean_t_cross),np.max(t_predict)])+0.2
t1=np.linspace(0,valmax,10)
fig=plt.figure(2)
plt.clf()
fig, ax = plt.subplots(1, 1)
ax.plot(mean_t_cross,t_predict,'bo',t1,t1,'r-')
# x=np.linspace(0,valmax,10)
# plt.plot(t_predict1,mean_t_cross,'ro',x,x,'k-')

ax.set_title('Mean zero crossing time for v(0)>0',fontsize=12)
plt.xlabel('Measured',fontsize=12)
plt.ylabel('Predicted',fontsize=12)
ax.legend(['Experiment','Ideal'],fontsize=12)
ax.set(xlim=(0, valmax), ylim=(0, valmax))
ax.set_aspect('equal')
fig.tight_layout()
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
if (not(to_plot_histogram)):
    plt.savefig('Predicted_vs_measured_zero_crossing.pdf')

# plt.hist(t_cross,bins=50, density=True)
# plt.xlabel("Time, first crossing v=0")
# plt.ylabel("Frequency")

# valmax=5
# #plt.plot(mean_t_cross,t_predict1,'ro',mean_t_cross,t_predict2,'bx')
# x=np.linspace(0,valmax,10)
# plt.plot(t_predict1,t_cross,'ro',x,x,'k-')
# plt.title('Mean zero crossing time for v(0)<0')
# plt.ylabel('Measured')
# plt.xlabel('Predicted')

# plt.xlim([0,valmax])
# plt.ylim([0,valmax])
# ax.set_aspect('equal')
# plt.savefig('Predicted_vs_measured_zero_crossing_all.pdf')

