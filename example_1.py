'''


NOTES:

For more sophisticated A&M style

'''

import numpy as np # for efficient array numerical operations 
import matplotlib.pylab as plt # for plotting
import seaborn as sns # for good colourblind colour scheme
import yaml # for reading in the input parameter YAML file

from atm_module import hypsometric, visc_mixture

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
  Tl[:] = Parmentier_T_p()
else:
  print('Invalid T structure selection')
  quit()

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
qv0 = param['qv0']
rho_d = param['rho_d']
ncld = len(cld_sp)

al = param['al']
fsed = param['fsed']
sigma = param['sigma']
alpha = param['alpha']

# We now have everything to perform the A&M (2001) calculation
# Loop over for each condensate calculation
qv = np.zeros((nlay,ncld))
qc = np.zeros((nlay,ncld))
qt = np.zeros((nlay,ncld))
qs = np.zeros((nlay,ncld))
rw = np.zeros((nlay,ncld))
rm = np.zeros((nlay,ncld))
nc = np.zeros((nlay,ncld))
for n in range(ncld):
  qv[:,n], qc[:,n], qt[:,n], qs[:,n], rw[:,n], rm[:,n], nc[:,n]  = \
    AandM_2001(nlay, qv0[n], cld_sp[n], fsed, al, sigma, alpha, rho_d[0], cld_mw[n], grav, altl, Tl, pl, Hp, Kzz, mu, eta, rho, cT)


fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

plt.plot(Tl,pl/1e6,c=colour[0],ls='solid',lw=2)
plt.yscale('log')
plt.gca().invert_yaxis()


fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

plt.plot(qv,pl/1e6,c=colour[0],ls='solid',lw=2,label='qv')
plt.plot(qc,pl/1e6,c=colour[1],ls='solid',lw=2,label='qc')
plt.plot(qt,pl/1e6,c=colour[2],ls='dotted',lw=2,label='qt')
plt.plot(qs,pl/1e6,c=colour[3],ls='dashed',lw=2,label='qs')

plt.legend()

plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()

fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

plt.plot(nc,pl/1e6,c=colour[0],ls='solid',lw=2,label='nc')

plt.legend()

plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()


fig = plt.figure() # Start figure 

colour = sns.color_palette('colorblind') # Decent colourblind wheel (10 colours)

plt.plot(rw*1e4,pl/1e6,c=colour[0],ls='solid',lw=2,label='rw')
plt.plot(rm*1e4,pl/1e6,c=colour[1],ls='solid',lw=2,label='rm')

plt.legend()

plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()

plt.show()