'''

In this scheme we use q in mass ratio units [g g^-1], this makes it easier to compare to other schemes.
'''

import numpy as np
from atm_module import vapour_pressure

Scloud = 0.0
R = 8.31446261815324e7

def AandM_2001(nlay, vap_VMR, vap_mw, cld_sp, fsed, sigma, alpha, rho_d, cld_mw, grav, altl, Tl, pl, met, al, Hp, Kzz, mu, eta, rho, cT):


  # Step through atmosphere to calculate condensate fraction at each layer
  # using A&M 2001 Eq. (8)

  q_v = np.zeros(nlay)
  q_c = np.zeros(nlay)
  q_t = np.zeros(nlay)
  q_s = np.zeros(nlay)

  # Set lowest boundary values - convert vapour VMR to MMR
  q_c[-1] = 0.0
  q_v[-1] = vap_VMR * vap_mw/mu[-1]
  q_t[-1] = q_v[-1]
  p_vap, q_s[-1] = vapour_pressure(vap_mw, cld_sp, Tl[-1], pl[-1], rho[-1], met)

  # Set through atmosphere from lower to upper boundary and perform calculation
  for k in range(nlay-2,-1,-1):
    # First calculate saturation mmr of species (qs) at layer
    p_vap, q_s[k] = vapour_pressure(vap_mw, cld_sp, Tl[k], pl[k], rho[k], met)
    # If the condensate mmr vapour is > than saturation mmr then it 
    # condenses in this layer, else it does not
    if (q_v[k+1] <= q_s[k]):
      # No condensation - solution is not changed from layer below
      q_v[k] = q_v[k+1]
      q_c[k] = 0.0
      q_t[k] = q_v[k] + q_c[k]
      continue
    else:
      # Condensation is triggered - adjust the condensate mixing ratio
      # according to equilibrium solution
      
      # Use A&M Eq. (7) to get total mixing ratio in layer
      dz = altl[k] - altl[k+1]
      L = al * (Hp[k] + Hp[k+1])/2.0
      #q_t[k] = q_t[k+1] * np.exp(-fsed * dz / L)

      q_t[k] = q_s[k] + (q_t[k+1] - q_s[k]) * np.exp(-fsed * dz / L)

      # Use A&M Eq. (8) to get condensate fraction
      q_c[k] = np.maximum(0.0,q_t[k] - (Scloud + 1.0)*q_s[k])

      # Account for any differences between vapour and condensate molecular weight
      q_c[k] = q_c[k]

      q_v[k] = q_t[k] - q_c[k] # (should be = qs[k] by definition)

  # We now have the condensation profile, next we use
  # the balance equation to estimate the cloud properties
  # The upward diffusion of total condensate must equal the downward
  # velocity of the condensate
 
  # Here we do a `cheat method' to quickly get a solution through assuming the particles are
  # in the Epstein drag regime (Kn >> 1).
  # Find the vertical convective velocity using Kzz = w * Hp (Marley & Robinson 2015)
  w = np.zeros(nlay)
  w[:] = Kzz[:]/(al*Hp[:])

  r_w = np.zeros(nlay)
  r_m = np.zeros(nlay)
  r_eff = np.zeros(nlay)
  N_c = np.zeros(nlay)
  for k in range(nlay):
    if (q_c[k] < 1e-10):
      # If low condensate fraction, assume zero
      r_w[k] = 0.0
      r_m[k] = 0.0
      N_c[k] = 0.0
    else:

      # Target radius of particle when settling velocity = fsed * w at each layer
      r_w[k] = (fsed*w[k]*2.0*cT[k]*rho[k])/(np.sqrt(np.pi)*grav*rho_d)

      # Median particle radius given log-normal distribution
      r_m[k] = r_w[k] * fsed**(1.0/alpha) * np.exp(-(alpha+6.0)/2.0 * np.log(sigma)**2)

      # Effective particle radius given log-normal distribution
      r_eff[k] = r_w[k] * fsed**(1.0/alpha) * np.exp(-(alpha+1.0)/2.0 * np.log(sigma)**2)

      # Total number density given log-normal distribution
      N_c[k] = (3.0 * q_c[k] * rho[k])/(4.0*np.pi*rho_d*r_m[k]**3) \
        * np.exp(-9.0/2.0 * np.log(sigma)**2)
 
  return q_v, q_c, q_t, q_s, r_w, r_m, r_eff, N_c