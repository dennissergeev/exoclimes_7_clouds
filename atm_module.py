'''

'''

import numpy as np

# Some physical constants
R = 8.31446261815324e7
kb = 1.380649e-16
amu = 1.66053906892e-24
bar = 1.0e6 # bar to dyne

# Coefficient parameters for Freedman et al. (2014) table fit
onedivpi = 1.0/np.pi
c1 = 10.602
c2 = 2.882
c3 = 6.09e-15
c4 = 2.954
c5 = -2.526
c6 = 0.843
c7 = -5.490
c8_l = -14.051; c8_h = 82.241
c9_l = 3.055; c9_h = -55.456
c10_l = 0.024; c10_h = 8.754
c11_l = 1.877; c11_h = 0.7048
c12_l = -0.445; c12_h = -0.0414
c13_l = 0.8321; c13_h = 0.8321

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

def k_Ross_Freedman(Tin, Pin, met):

  # Calculates the IR band Rosseland mean opacity (local T) according to the
  # Freedman et al. (2014) fit and coefficients

  # Input:
  # T - Local gas temperature [K]
  # P - Local gas pressure [dyne]
  # met - Local metallicity [M/H] (log10 from solar, solar [M/H] = 0.0)

  # Output:
  # k_IR - IR band Rosseland mean opacity [cm2 g-1]

  T = Tin
  P = Pin

  Tl10 = np.log10(T)
  Pl10 = np.log10(P)

  # Low pressure expression
  k_lowP = c1*np.arctan(Tl10 - c2) \
    - (c3/(Pl10 + c4))*np.exp((Tl10 - c5)**2) \
    + c6*met + c7

  # Temperature split for coefficients = 800 K
  if (T <= 800.0):
    k_hiP = c8_l + c9_l*Tl10 \
      + c10_l*Tl10**2 + Pl10*(c11_l + c12_l*Tl10) \
      + c13_l * met * (0.5 + onedivpi*np.arctan((Tl10 - 2.5) / 0.2))
  else:
    k_hiP = c8_h + c9_h*Tl10 \
      + c10_h*Tl10**2 + Pl10*(c11_h + c12_h*Tl10) \
      + c13_h * met * (0.5 + onedivpi*np.arctan((Tl10 - 2.5) / 0.2))

  # Total Rosseland mean opacity
  k_IR = (10.0**k_lowP + 10.0**k_hiP)

  # Avoid divergence in fit for large values
  k_IR = np.minimum(k_IR,1.0e30)

  return k_IR

def adiabat_correction(nlay,Tl,pl,kappa):
  # Subroutine that corrects for adiabatic region following Parmentier & Guillot (2015)
  # But here we correct using kappa rather than the gradient expression from Parmentier & Guillot (2015)

  gradrad = np.zeros(nlay)
  gradad = np.zeros(nlay)
  for k in range(nlay-1):
    gradrad[k] = np.log(Tl[k]/Tl[k+1])/np.log(pl[k]/pl[k+1])
    gradad[k] = kappa

  gradrad[-1] = 0.0
  gradad[-1] = 0.0

  iRC = nlay-1
  prc = pl[-1]

  for k in range(nlay-1, 0, -1):
    if (gradrad[k] > gradad[k]):
      iRC = k
      prc = pl[iRC]

  print('RC boundary: ', prc/1e6)

  if (iRC < nlay):
    for k in range(iRC, nlay-1, 1):
      gradad[k] = kappa
      Tl[k+1] = Tl[k] * (pl[k+1]/pl[k])**gradad[k]

  return Tl