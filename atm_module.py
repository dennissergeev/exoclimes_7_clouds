'''

'''

import numpy as np

R = 8.31446261815324e7
kb = 1.380649e-16
amu = 1.66053906892e-24
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

def q_s_sat(vap_mw, cld_sp, T, p, rho, met):

  # Calculate vapour pressure of species
  if (cld_sp == 'MgSiO3'):
    # Visscher - taken from VIRGA
    p_vap = 10.0**(13.43 - 28665.0/T - met) * bar


  # Specific gas constant of condensable vapour 
  Rd_v = R/vap_mw

  # Saturation mass mixing ratio
  q_s = (p_vap/(Rd_v * T))/rho

  return q_s

def visc_mixture(T, nbg, bg_VMR, bg_mw, bg_d, bg_LJ):

  # Davidson (1993) dynamical viscosity mixing rule

  # First calculate each species eta
  eta_g = np.zeros(nbg)
  for n in range(nbg):
    eta_g[n] = (5.0/16.0) * (np.sqrt(np.pi*(bg_mw[n]*amu)*kb*T)/(np.pi*bg_d[n]**2)) \
      * ((((kb*T)/(kb*bg_LJ[n]))**(0.16))/1.22)

  # Calculate y values
  y = np.zeros(nbg)
  bot = 0.0
  for n  in range(nbg):
    bot = bot + bg_VMR[n] * np.sqrt(bg_mw[n])
  y[:] = (bg_VMR[:] * np.sqrt(bg_mw[:]))/bot

  # Calculate fluidity following Davidson equation
  eta = 0.0
  for i in range(nbg):
    for j in range(nbg):
      Eij = ((2.0*np.sqrt(bg_mw[i]*bg_mw[j]))/(bg_mw[i] + bg_mw[j]))**0.375
      part = (y[i]*y[j])/(np.sqrt(eta_g[i]*eta_g[j])) * Eij
      eta = eta + part

  # Viscosity is inverse fluidity
  eta = 1.0/eta

  return eta