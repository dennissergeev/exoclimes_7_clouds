'''


NOTES:

For more sophisticated A&M style

'''

import numpy as np # for efficient array numerical operations 
import matplotlib.pylab as plt # for plotting
import seaborn as sns # for good colourblind colour scheme
import yaml # for reading in the input parameter YAML file

from atm_module import hypsometric, visc_mixture, adiabat_correction

from T_p_Guillot_2010 import Guillot_T_p # Import function for Guillot 2010 semi-grey profile
from T_p_Parmentier_2015 import Parmentier_T_p # Import function for Parmentier & Guillot 2015 picket-fence profile

from AandM_2001 import AandM_2001

kb = 1.380649e-16
amu = 1.66053906892e-24

# Open parameter YAML file and read parameters for A&M profile
with open('parameters.yaml', 'r') as file:
  param = yaml.safe_load(file)['A&M']

# Now extract the parameters from the YAML file into local variables

# Give number of layers in 1D atmosphere - number of levels is nlay + 1
nlay = param['nlay']
nlev = nlay + 1

mu_z = param['mu_z']
Tirr = param['Tirr']
Tint = param['Tint']
k_v =  param['k_v']
k_ir = param['k_ir']
met = param['met']
grav = param['grav']
kappa = param['kappa']

# Get top and bottom pressure layers in bar - convert to dyne
ptop = param['ptop'] * 1e6
pbot = param['pbot'] * 1e6

# Get pressure at levels (edges) - assume log-spaced
pe = np.logspace(np.log10(ptop),np.log10(pbot),nlev)

# Get pressure at layers using level spacing
pl = np.zeros(nlay)
pl[:] = (pe[1:] - pe[0:-1])/np.log(pe[1:]/pe[0:-1])
 

# Kzz profile - here we assume constant but a 1D array for easy change to
# non-constant or some function with some editing
# A&M (2001) use a convective heat flux mixing length theory prescription
Kzz = np.zeros(nlay)
Kzz[:] = param['Kzz']
# Example change - use expression from Parmentier et al. (2013):
#Kzz[:] = 5e8/np.sqrt(pl[:]*1e-6)

# We do the same for molecular weight of the atmosphere 
# Can be changed to varying with height/pressure
mu = np.zeros(nlay)
mu[:] = param['mu']

# Get 1D analytical temperature pressure profile at layers
Tl = np.zeros(nlay)
if (param['Guillot'] == True):
  Tl[:] = Guillot_T_p(nlay, pbot, pl, k_v, k_ir, Tint, mu_z, Tirr, grav)
elif (param['Parmentier'] == True):
  Tl[:] = Parmentier_T_p(nlay, pl, Tint, mu_z, Tirr, grav, met, 1)
else:
  print('Invalid T structure selection')
  quit()

if (param['adibat_corr'] == True):
  Tl[:] = adiabat_correction(nlay, Tl, pl, kappa)

# Atmosphere mass density
rho = np.zeros(nlay)
rho[:] = (pl[:]*mu[:]*amu)/(kb * Tl[:])

# Atmosphere thermal velocity
cT = np.zeros(nlay)
cT[:] = np.sqrt((2.0 * kb * Tl[:]) / (mu[:] * amu))

# Find the altitude grid and scale heights at levels and layers using the hypsometric equation
alte = np.zeros(nlev)
alte, Hp = hypsometric(nlev, Tl, pe, mu, grav)
altl = (alte[0:-1] + alte[1:])/2.0

# Volume mixing ratio of condensate vapour at lower boundary
# Deep abyssal mixing ratio (assume infinite supply at constant value)
bg_sp = param['bg_sp']
bg_VMR = param['bg_VMR']
bg_mw = param['bg_mw']
bg_d = param['bg_d']
bg_LJ = param['bg_LJ']
nbg = len(bg_sp)

# Find dynamical viscosity of each layer given a background gas mixture
eta = np.zeros(nlay)
for k in range(nlay):
  eta[k] = visc_mixture(Tl[k], nbg, bg_VMR, bg_mw, bg_d, bg_LJ)

cld_sp = param['cld_sp']
cld_mw = param['cld_mw']
rho_d = param['rho_d']
vap_mw =  param['vap_mw']
vap_VMR = param['vap_VMR']

ncld = len(cld_sp)

al = param['al']
fsed = param['fsed']
sigma = param['sigma']
alpha = param['alpha']

# We now have everything to perform the A&M (2001) calculation
# Loop over for each condensate calculation
q_v = np.zeros((nlay,ncld))
q_c = np.zeros((nlay,ncld))
q_t = np.zeros((nlay,ncld))
q_s = np.zeros((nlay,ncld))
r_w = np.zeros((nlay,ncld))
r_m = np.zeros((nlay,ncld))
r_eff = np.zeros((nlay,ncld))
N_c = np.zeros((nlay,ncld))
for n in range(ncld):
  q_v[:,n], q_c[:,n], q_t[:,n], q_s[:,n], r_w[:,n], r_m[:,n], r_eff[:,n], N_c[:,n]  = \
    AandM_2001(nlay, vap_VMR[n], vap_mw[n], cld_sp[n], fsed[n], sigma[n], alpha[n], rho_d[n], cld_mw[n], grav, altl, Tl, pl, met, al, Hp, Kzz, mu, eta, rho, cT)


fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

plt.plot(Tl,pl/1e6,c=colour[0],ls='solid',lw=2,label=r'T-p')
plt.xlabel(r'$T$ [K]',fontsize=16)
plt.ylabel(r'$p$ [bar]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend()
plt.yscale('log')
plt.gca().invert_yaxis()
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

for n in range(ncld):
  if (n == 0):
    plt.plot(q_v[:,n],pl/1e6,c=colour[n],ls='solid',lw=2,label=r'$q_{\rm v}$')
    plt.plot(q_c[:,n],pl/1e6,c=colour[n],ls='dashed',lw=2,label=r'$q_{\rm c}$')
    plt.plot(q_t[:,n],pl/1e6,c=colour[n],ls='dashdot',lw=2,label=r'$q_{\rm t}$')
    plt.plot(q_s[:,n],pl/1e6,c=colour[n],ls='dotted',lw=2,label=r'$q_{\rm s}$')
  else:
    plt.plot(q_v[:,n],pl/1e6,c=colour[n],ls='solid',lw=2)
    plt.plot(q_c[:,n],pl/1e6,c=colour[n],ls='dashed',lw=2)
    plt.plot(q_t[:,n],pl/1e6,c=colour[n],ls='dashdot',lw=2)
    plt.plot(q_s[:,n],pl/1e6,c=colour[n],ls='dotted',lw=2)    


plt.xlabel(r'$q$ [g g$^{-1}$]',fontsize=16)
plt.ylabel(r'$p$ [bar]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

for n in range(ncld):
  if (n == 0):
    plt.plot(N_c[:,n],pl/1e6,c=colour[n],ls='solid',lw=2,label=r'$N_{\rm c}$')
  else:
    plt.plot(N_c[:,n],pl/1e6,c=colour[n],ls='solid',lw=2)
plt.xlabel(r'$N_{\rm c}$ [cm$^{-3}$]',fontsize=16)
plt.ylabel(r'$p$ [bar]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

for n in range(ncld):
  if (n == 0):
    plt.plot(r_w[:,n]*1e4,pl/1e6,c=colour[n],ls='solid',lw=2,label=r'$r_{\rm w}$')
    plt.plot(r_m[:,n]*1e4,pl/1e6,c=colour[n],ls='dashed',lw=2,label=r'$r_{\rm m}$')
    plt.plot(r_eff[:,n]*1e4,pl/1e6,c=colour[n],ls='dotted',lw=2,label=r'$r_{\rm eff}$')
  else:
    plt.plot(r_w[:,n]*1e4,pl/1e6,c=colour[n],ls='solid',lw=2)
    plt.plot(r_m[:,n]*1e4,pl/1e6,c=colour[n],ls='dashed',lw=2)   
    plt.plot(r_eff[:,n]*1e4,pl/1e6,c=colour[n],ls='dotted',lw=2)
plt.xlabel(r'$r$ [$\mu$m]',fontsize=16)
plt.ylabel(r'$p$ [bar]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend()
plt.xlim(1e-3,1e3)
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.show()