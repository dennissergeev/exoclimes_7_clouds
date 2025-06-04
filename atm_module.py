'''

'''

import numpy as np

# Some physical constants
R = 8.31446261815324e7
kb = 1.380649e-16
amu = 1.66053906892e-24

#Conversions to dyne
bar = 1.0e6 # bar to dyne
atm = 1.01325e6 # atm to dyne
pa = 10.0 # pa to dyn
mmHg = 1333.22387415  # mmHg to dyne

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

  # Calculate vapour pressure of species in dyne
  match cld_sp:
    case 'C':
      # Gail & Sedlmayr (2013) - I think...
      p_vap = np.exp(3.27860e1 - 8.65139e4/(T + 4.80395e-1))
    case 'TiC':
      # Kimura et al. (2023)
      p_vap = 10.0**(-33600.0/T + 7.652) * atm
    case 'SiC':
      # Elspeth 5 polynomial JANAF-NIST fit
      p_vap =  np.exp(-9.51431385e4/T + 3.72019157e1 + 1.09809718e-3*T \
        - 5.63629542e-7*T**2 + 6.97886017e-11*T**3)
    case 'CaTiO3':
      # Wakeford et al. (2017) -  taken from VIRGA
      p_vap = 10.0**(-72160.0/T + 30.24 - np.log10(p/1e6) - 2.0*met) * bar
    case 'TiO2':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-7.70443e4/T +  4.03144e1 - 2.59140e-3*T \
        + 6.02422e-7*T**2 - 6.86899e-11*T**3)
    case 'VO':
      # NIST 5 param fit
      p_vap = np.exp(-6.74603e4/T + 3.82717e1 - 2.78551e-3*T \
        + 5.72078e-7*T**2 - 7.41840e-11*T**3)
    case 'Al2O3':
      # Wakeford et al. (2017) - taken from CARMA
      p_vap = 10.0**(17.7 - 45892.6/T - 1.66*met) * bar
    case 'Fe':
      # Visscher et al. (2010) - taken from CARMA
      p_vap = 10.0**(7.23 - 20995.0/T) * bar
    case 'FeS':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-5.69922e4/T + 3.86753e1 - 4.68301e-3*T \
        + 1.03559e-6*T**2 - 8.42872e-11*T**3)
    case 'FeO':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-6.30018e4/T + 3.66364e1 - 2.42990e-3*T \
        + 3.18636e-7*T**2)
    case 'Mg2SiO4':
      # Visscher et al. (2010)/Visscher notes - taken from CARMA
      p_vap = 10.0**(14.88 - 32488.0/T - 1.4*met - 0.2*np.log10(p/1e6)) * bar
    case 'MgSiO3':
      # Visscher - taken from VIRGA
      p_vap = 10.0**(13.43 - 28665.0/T - met) * bar
    case 'MgO':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-7.91838e4/T + 3.57312e1 + 1.45021e-4*T \
        - 8.47194e-8*T**2 + 4.49221e-12*T**3)
    case 'SiO2':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-7.28086e4/T + 3.65312e1 - 2.56109e-4*T \
        - 5.24980e-7*T**2 + 1.53343E-10*T**3) 
    case 'SiO':
      # Gail et al. (2013)
      p_vap = np.exp(-49520.0/T + 32.52)
    case 'Cr':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-4.78455e+4/T + 3.22423e1 - 5.28710e-4*T \
        - 6.17347e-8*T**2 + 2.88469e-12*T**3)
    case 'MnS':
      # Morley et al. (2012)
      p_vap = 10.0**(11.532 - 23810.0/T - met) * bar
    case 'Na2S':
      # Morley et al. (2012)
      p_vap =  10.0**(8.550 - 13889.0/T - 0.5*met) * bar
    case 'ZnS':
      # Elspeth 5 polynomial Barin data fit
      p_vap = np.exp(-4.75507888e4/T + 3.66993865e1 - 2.49490016e-3*T \
        + 7.29116854e-7*T**2 - 1.12734453e-10*T**3)       
    case 'KCl':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-2.69250e4/T + 3.39574e+1 - 2.04903e-3*T \
        - 2.83957e-7*T**2 + 1.82974e-10*T**3)
    case 'NaCl':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-2.79146e4/T + 3.46023e1 - 3.11287e3*T \
        + 5.30965e-7*T**2 -2.59584e-12*T**3)
    case 'S2':
      #--- Zahnle et al. (2016) ---
      if (T < 413.0):
        p_vap = np.exp(27.0 - 18500.0/T) * bar
      else:
        p_vap = np.exp(16.1 - 14000.0/T) * bar
    case 'S8':
      #--- Zahnle et al. (2016) ---
      if (T < 413.0):
        p_vap = np.exp(20.0 - 11800.0/T) * bar
      else:
        p_vap = np.exp(9.6 - 7510.0/T) * bar       
    case 'NH4Cl':
      # Unknown - I think I fit this?
      p_vap = 10.0**(7.0220 - 4302.0/T) * bar
    case 'H2O':
      TC = T - 273.15
      # Huang (2018) - A Simple Accurate Formula for Calculating Saturation Vapor Pressure of Water and Ice
      if (TC < 0.0):
        f = 0.99882 * exp(0.00000008 * p/pa)
        p_vap = np.exp(43.494 - (6545.8/(TC + 278.0)))/(TC + 868.0)**2.0 * pa * f
      else:
        f = 1.00071 * exp(0.000000045 * p/pa)
        p_vap = np.exp(34.494 - (4924.99/(TC + 237.1)))/(TC + 105.0)**1.57 * pa * f
    case 'NH3':
      # Blakley et al. (2024) - experimental to low T and pressure
      p_vap = exp(-5.55 - 3605.0/T + 4.82792*np.log(T) - 0.024895*T + 2.1669e-5*T**2 - 2.3575e-8 *T**3) * bar
    case 'CH4':
      # Frey & Schmitt (2009)
      p_vap = exp(1.051e1 - 1.110e3/T - 4.341e3/T**2 + 1.035e5/T**3 - 7.910e5/T**4) * bar
    case 'NH4SH':
      #--- E.Lee's fit to Walker & Lumsden (1897) ---
      p_vap = 10.0**(7.8974 - 2409.4/T) * bar
    case 'H2S':
      # Frey & Schmitt (2009)
      p_vap = np.exp(12.98 - 2.707e3/T) * bar
    case 'H2SO4':
      # GGChem 5 polynomial NIST fit
      p_vap = np.exp(-1.01294e4/T + 3.55465e1 - 8.34848e-3*T)      
    case 'CO':
      # Frey & Schmitt (2009)
      if (T < 61.55):
        p_vap = np.exp(1.043e1 - 7.213e2/T - 1.074e4/T**2 + 2.341e5/T**3 - 2.392e6/T**4 + 9.478e6/T**5) * bar
      else:
        p_vap = np.exp(1.025e1 - 7.482e2/T - 5.843e3/T**2 + 3.939e4/T**3) * bar
    case 'CO2':
      # Frey & Schmitt (2009)
      if (T < 194.7):
        p_vap = np.exp(1.476e1 - 2.571e3/T - 7.781e4/T**2 + 4.325e6/T**3 - 1.207e8/T**4 + 1.350e9/T**5) * bar
      else:
        p_vap = np.exp(1.861e1 - 4.154e3/T + 1.041e5/T**2) * bar
    case 'O2':
      # Blakley et al. (2024) - experimental to low T and pressure (beta O2)
      p_vap = np.exp(15.29 - 1166.2/T - 0.75587*np.log(T) + 0.14188*T - 1.8665e-3*T**2 + 7.582e-6 *T**3) * bar
    case _:
      print('Vapour pressure species not found: ', cld_sp)
      print('quitting')
      quit()


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