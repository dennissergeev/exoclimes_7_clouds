'''

'''

import numpy as np
from atm_module import vapour_pressure, surface_tension
from scipy.integrate import solve_ivp

kb = 1.380649e-16
R = 8.31446261815324e7
amu = 1.66053906892e-24
Avo = 6.02214076e23

r_seed = 1e-7
V_seed = 4.0/3.0 * np.pi * r_seed**3

def calc_cond(ncld, r_c, Kn, n_v, D, vth, sat, m0, V_mix):


  dmdt = np.zeros(ncld)

  for n in range(ncld):

    # # Diffusive limited regime (Kn << 1) [g s-1]
    # dmdt_low = 4.0 * np.pi * r_c * D[n] * m0[n] * n_v[n] \
    #   *  (1.0 - 1.0/sat[n])

    # # Free molecular flow regime (Kn >> 1) [g s-1]
    dmdt_high = 4.0 * np.pi * r_c**2 * vth[n] * m0[n] * n_v[n] * 1.0 \
      * (1.0 - 1.0/sat[n]) * V_mix[n]

    # # If evaporation, weight rate by current condensed volume ratio (Woitke et al. 2020)
    # if (sat[n] < 1.0):
    #   dmdt_high = dmdt_high * V_mix[n]
    #   dmdt_low = dmdt_low * V_mix[n]

    # # Critical Knudsen number
    # Kn_crit = Kn * (dmdt_high/dmdt_low)

    # # Kn' (Woitke & Helling 2003)
    # Knd = Kn/Kn_crit

    # # tanh interpolation function
    # fx = 0.5 * (1.0 - np.tanh(2.0*np.log10(Knd)))

    # Mass change rate
    dmdt[n] = dmdt_high #dmdt_low * fx + dmdt_high * (1.0 - fx)

  return dmdt

def calc_hom_nuc(ncld, n_v, T, p, mw_cld, rho_d, cld_nuc, sig_inf, sat, r0):

  # For simplicity, we use the CARMA classical homogenous nucleation rate expression here
  # not modified homogenous nucleation theory

  J_hom = np.zeros(ncld)

  for n in range(ncld):

    if (cld_nuc[n] == False):
      J_hom[n] = 0.0
      continue

    if (sat[n] <= 1.0):
      J_hom[n] = 0.0
    else:

      alpha = 1.0
      Nf = 5.0
      third = 1.0/3.0
      twothird = 2.0/3.0

      ln_ss = np.log(sat[n]) 
      f0 = 4.0 * np.pi * r0[n]**2 
      kbT = kb * T       

      theta_inf = (f0 * sig_inf[n])/(kbT)  
      N_inf = (((2.0/3.0) * theta_inf) / ln_ss)**3

      N_star = 1.0 + (N_inf / 8.0) \
        * (1.0 + np.sqrt(1.0 + 2.0*(Nf/N_inf)**third) \
        - 2.0*(Nf/N_inf)**third)**3
      N_star = np.maximum(1.00001, N_star)
      N_star_1 = N_star - 1.0

      dg_rt = theta_inf * (N_star_1 / (N_star_1**third + Nf**third))

      Zel = np.sqrt((theta_inf / (9.0 * np.pi * (N_star_1)**(4.0/3.0))) \
        * ((1.0 + 2.0*(Nf/N_star_1)**1.0/3.0)/(1.0 + (Nf/N_star_1)**third)**3))

      tau_gr = (f0 * N_star**twothird) * alpha * np.sqrt(kbT \
        / (2.0 * np.pi * mw_cld[n] * amu)) * n_v[n]

      J_hom[n] = n_v[n] * tau_gr * Zel * np.exp(np.maximum(-300.0, N_star_1*ln_ss - dg_rt))

  return J_hom

def calc_seed_evap(ncld, N_c, m_c, m_seed, sat, cld_nuc):

  J_evap = np.zeros(ncld)

  for n in range(ncld):

    if (cld_nuc[n] == False):
      J_evap[n] = 0.0
      continue

    if (sat[n] >= 1.0):
      # If saturated then evaporation can't take place
      J_evap[n] = 0.0
    else:
      # Check if average mass is around 0.1% the seed particle mass
      # This means the core is (probably) exposed to the air and can evaporate freely
      if (m_c <= (1.001 * m_seed[n])):
        tau_evap = 0.1
        # Seed particle evaporation rate [cm-3 s-1]
        J_evap[n] = -N_c/tau_evap
      else:
        # There is still some mantle to evaporate from
        J_evap[n] = 0.0

  return J_evap

def dqdt(t, y, ncld, p_vap, vth, D, sig_inf, Rd_v, cld_nuc, nd_atm, rho, mfp, T, p, m_seed, rho_d, m0, cld_mw, r0):

  # Limit y values
  y[:] = np.maximum(y[:],1e-30)

  f = np.zeros(len(y))

  rho_c = np.zeros(ncld)
  rho_v = np.zeros(ncld)
  p_v = np.zeros(ncld)
  n_v = np.zeros(ncld)
  sat = np.zeros(ncld)
  V_mix = np.zeros(ncld)

  # Convert y to real physical numbers to calculate f
  N_c = y[0]*nd_atm
  rho_c[:] = y[1:2+ncld-1]*rho 
  rho_v[:] = y[2+ncld-1:]*rho 

  # Find the partial pressure of the vapour and number density
  p_v[:] = rho_v[:] * Rd_v[:] * T
  n_v[:] = p_v[:]/(kb*T) 

  # Find supersaturation ratio
  sat[:] = p_v[:]/p_vap[:]

  # Total condensed mass [g cm^-3]
  rho_c_t = np.sum(rho_c[:])

  # Mean mass of particle [g]
  m_c = np.maximum(rho_c_t/N_c, m_seed[0])

  # Bulk density of particle mixture [g cm^-3]
  rho_d_m = 0.0
  for n in range(ncld):
    rho_d_m += (rho_c[n]/rho_c_t) * rho_d[n]

  # Mass weighted mean radius of particle [cm]
  r_c = np.maximum(((3.0*m_c)/(4.0*np.pi*rho_d_m))**(1.0/3.0), r_seed)

  # Bulk material volume mixing ratio of mixture
  V_tot = np.sum(rho_c[:]/rho_d[:]) # Total condensed volume
  V_mix[:] = (rho_c[:]/rho_d[:])/V_tot # Condensed volume mixing ratio

  # Knudsen number
  Kn = mfp/r_c

  # Calculate condensation/evaporation rate
  f_cond = calc_cond(ncld, r_c, Kn, n_v, D, vth, sat, m0, V_mix)

  # Calculate homogenous nucleation rate
  f_nuc_hom = calc_hom_nuc(ncld, n_v, T, p, cld_mw, rho_d, cld_nuc, sig_inf, sat, r0)

  # Calculate seed particle evaporation rate
  f_seed_evap = calc_seed_evap(ncld, N_c, m_c, m_seed, sat, cld_nuc)

  # Here you could also calculate coagulation and coalescence collisional growth rates -
  # but not included in this example - but we keep dummy variables here so you can see the form and scaling of the equations
  f_coag = 0.0
  f_coal = 0.0

  # Calculate final net flux rate for each moment and vapour
  f[0] = np.sum(f_nuc_hom[:] + f_seed_evap[:]) + (f_coag + f_coal)*N_c**2
  f[1:2+ncld-1] = m_seed[:]*(f_nuc_hom[:]  + f_seed_evap[:]) + f_cond[:]*N_c
  f[2+ncld-1:] = -f[1:2+ncld-1]

  # Convert f to ratios
  f[0] = f[0]/nd_atm
  f[1:] = f[1:]/rho

  # Check if condensation from vapour is viable
  for n in range(ncld):
    if (rho_v[n]/rho <= 1e-28):
      if (f[1+n] > 0.0):
        f[1+n] = 0.0
        f[1+ncld+n] = 0.0

  # Check if evaporation from condensate is viable
  for n in range(ncld):
    if (rho_c[n]/rho <= 1e-28):
      if (f[1+ncld+n] > 0.0):
        f[1+n] = 0.0
        f[1+ncld+n] = 0.0

  return f

def two_moment_mixed(nlay, ncld, t_step, vap_VMR, vap_mw, cld_sp, rho_d, cld_mw, Tl, pl, nd_atm, rho, mfp, mu, met, cld_nuc, q_v, q_0, q_1):

  # work arrays
  p_vap = np.zeros(ncld)
  q_s = np.zeros((nlay, ncld))
  vth = np.zeros(ncld)
  D = np.zeros(ncld)
  sig_inf = np.zeros(ncld)
  Rd_v = np.zeros(ncld)

  m0 = np.zeros(ncld)
  V0 = np.zeros(ncld)
  r0 = np.zeros(ncld)
  d0 = np.zeros(ncld)
  m_seed = np.zeros(ncld)
  for n in range(ncld):
    m0[n] = cld_mw[n] * amu
    V0[n] = m0[n] / rho_d[n]
    r0[n] = ((3.0*V0[n])/(4.0*np.pi))**(1.0/3.0)
    d0[n] = 2.0 * r0[n]
    Rd_v[n] = R/vap_mw[n]
    if (cld_nuc[n] == False):
      m_seed[n] = 0.0
    else:
      m_seed[n] = V_seed * rho_d[n]

  # Prepare integration solver
  y0 = np.zeros(1+ncld+ncld) # one q_0, ncld q_1 and ncld q_v

  # Tolerances and time-stepping
  rtol = 1e-3
  atol = 1e-30
  max_step = np.inf
  t_span = [0.0, t_step]

  for k in range(nlay):

    # Calculate constant variables for each species
    for n in range(ncld):

      # Saturation vapour mass mixing ratio for this layer
      p_vap[n], q_s[k,n] = vapour_pressure(vap_mw[n], cld_sp[n], Tl[k], pl[k], rho[k], met)

      # Thermal velocity of vapour [cm s-1]
      vth[n] = np.sqrt((kb*Tl[k])/(2.0*np.pi*vap_mw[n]*amu))

      # Gaseous diffusion constant of vapour [cm2 s-1] 
      D[n] = 5.0/(16.0*Avo*d0[n]**2*rho[k]) * np.sqrt((R*Tl[k]*mu[k])/(2.0*np.pi) * (vap_mw[n] + mu[k])/vap_mw[n])

      # Surface tension of species [erg cm-2]
      sig_inf[n] = surface_tension(cld_sp[n], Tl[k])

    # Give tracer values to y array
    y0[0] = q_0[k]
    y0[1:2+ncld-1] = q_1[k,:]
    y0[2+ncld-1:] = q_v[k,:]

    # Use implicit stiff method to integrate the tracer values in time
    sol = solve_ivp(dqdt, t_span, y0, method='Radau', rtol=rtol, atol=atol, \
      args=(ncld, p_vap, vth, D, sig_inf, Rd_v, cld_nuc, nd_atm[k], rho[k], mfp[k], Tl[k], pl[k], m_seed, rho_d, m0, cld_mw, r0))

    # Give back results to the vapour and condensate array
    q_0[k] = sol.y[0,-1]
    q_1[k,:] = sol.y[1:2+ncld-1,-1]
    q_v[k,:] = sol.y[2+ncld-1:,-1]

    q_0[k] = np.maximum(q_0[k], 1e-30)
    for n in range(ncld):
      q_1[k,n] = np.maximum(q_1[k,n], 1e-30)
      q_v[k,n] = np.maximum(q_v[k,n], 1e-30)
      q_s[k,n] = np.maximum(q_s[k,n], 1e-30)

  return q_v, q_0, q_1, q_s