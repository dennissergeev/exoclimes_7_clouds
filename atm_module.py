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
  elif (cld_sp == 'Fe'):
    # Visscher et al. (2010) - taken from CARMA
    p_vap = 10.0**(7.23 - 20995.0/T) * bar

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


def v_f_sat_adj(nlay, r_m, sig, grav, rho_d, rho, eta, mfp, cT):


  # Calculate settling velocity v_f [cm s-1] at each layer
  v_f = np.zeros(nlay)
  for k in range(nlay):

    # Volume (or mass) weighted mean radius of particle assuming log-normal distribution
    r_c = np.maximum(r_m * np.exp(7.0/2.0 * np.log(sig)**2),1e-7)

    # Knudsen number
    Kn = mfp[k]/r_c
    Kn_b = np.minimum(Kn, 100.0)

    # Cunningham slip factor (Kim et al. 2005)
    beta = 1.0 + Kn_b*(1.165 + 0.483 * np.exp(-0.997/Kn_b))

    # Stokes regime (Kn << 1) settling velocity (Ohno & Okuzumi 2017)
    v_f_St = (2.0 * beta * grav * r_c**2 * (rho_d - rho[k]))/(9.0 * eta[k]) \
      * (1.0 + ((0.45*grav*r_c**3*rho[k]*rho_d)/(54.0*eta[k]**2))**(0.4))**(-1.25)

    # Epstein regime (Kn >> 1) regime settling velocity (Woitke & Helling 2003)
    v_f_Ep = (np.sqrt(np.pi)*grav*rho_d*r_c)/(2.0*cT[k]*rho[k])

    # tanh interpolation function for Kn ~ 1
    fx = 0.5 * (1.0 - np.tanh(2.0*np.log10(Kn)))

    # Interpolation for settling velocity
    v_f[k] = fx*v_f_St + (1.0 - fx)*v_f_Ep
    v_f[k] = np.maximum(v_f[k], 1e-30)

  return v_f