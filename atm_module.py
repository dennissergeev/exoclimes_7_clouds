'''

'''

import numpy as np

R = 8.31446261815324e7

bar = 1.0e6 # bar to dyne


def hypsometric(nlev, Tl, pe, mu, grav):

  alte = np.zeros(nlev)

  alte[-1] = 0.0
  for k in range(nlev-2,-1,-1):
    alte[k] = alte[k+1] + (R*Tl[k])/(mu[k]*grav) * np.log(pe[k+1]/pe[k])

  Hp = np.zeros(nlev-1)
  for k in range(nlev-1):
    Hp[k] = (R*Tl[k])/(mu[k]*grav)

  return alte, Hp

def sat_vmr(cld_sp, T, p):

  if (cld_sp == 'MgSiO3'):
    # Ackerman & Marley (2001)
    p_vap = np.exp(-58663.0/T + 25.37) * bar

  qs = np.minimum(1.0, p_vap/p)

  return qs