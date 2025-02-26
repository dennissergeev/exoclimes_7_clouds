'''

'''

import numpy as np
from atm_module import sat_vmr

Scloud = 0.0
R = 8.31446261815324e7

def AandM_2001(nlay, qv0, cld_sp, fsed, al, sigma, alpha, rho_d, mw_cld, grav, altl, Tl, pl, Hp, Kzz, mu, eta, rho, cT):


  # Step through atmosphere to calculate condensate fraction at each layer
  # using A&M 2001 Eq. (8)

  qv = np.zeros(nlay)
  qc = np.zeros(nlay)
  qt = np.zeros(nlay)
  qs = np.zeros(nlay)

  qc[-1] = 0.0
  qv[-1] = qv0
  qt[-1] = qv0
  qs[-1] = sat_vmr(cld_sp, Tl[-1], pl[-1])

  for k in range(nlay-2,-1,-1):
    # First calculate saturation vmr of species (qs) at layer
    qs[k] = sat_vmr(cld_sp, Tl[k], pl[k])
    # If the condensate vapour is > than saturation vmr then it 
    # condenses in this layer, else it does not
    if (qv[k+1] <= qs[k]):
      # No condensation - solution is not changed from layer below
      qv[k] = qv[k+1]
      qc[k] = 0.0
      qt[k] = qv[k] + qc[k]
      continue
    else:
      # Condensation is triggered - adjust the condensate mixing ratio
      # according to equilibrium solution
      
      # Use A&M Eq. (7) to get total mixing ratio in layer
      dz = altl[k] - altl[k+1]
      L = al * (Hp[k] + Hp[k+1])/2.0
      qt[k] = qt[k+1] * np.exp(-fsed * dz / L)

      # Use A&M Eq. (8) to get condensate fraction
      qc[k] = np.maximum(0.0,qt[k] - (Scloud + 1.0)*qs[k])

      qv[k] = qt[k] - qc[k] # (should be = qs[k] by definition)

  # We now have the condensation profile, next we use
  # the balance equation to estimate the cloud properties
  # The upward diffusion of total condensate must equal the downward
  # velocity of the condensate
 
  # Here we do a `cheat method' to quickly get a solution through assuming the particles are
  # in the Epstein drag regime.
  # Find the vertical convective velocity using Kzz = w * Hp (Marley & Robinson 2015)
  # Scale with assumed mixing length factor and fsed
  w = np.zeros(nlay)
  w[:] = Kzz[:]/(al*Hp[:])*fsed

  # Target settling velocity of particles must = w at each layer
  rw = np.zeros(nlay)
  rm = np.zeros(nlay)
  nc = np.zeros(nlay)
  for k in range(nlay):
    if (qc[k] < 1e-10):
      # If low condensate fraction, assume zero
      rw[k] = 0.0
      rm[k] = 0.0
      nc[k] = 0.0
    else:

      rw[k] = (w[k]*2.0*cT[k]*rho[k])/(np.sqrt(np.pi)*grav*rho_d)

      rm[k] = rw[k] * fsed**(1.0/alpha) * np.exp(-(alpha+6.0)/2.0 * np.log(sigma)**2)

      eps = mw_cld/mu[k]
      nc[k] = (3.0 * qc[k] * eps * rho[k])/(4.0*np.pi*rho_d*rm[k]**3) \
        * np.exp(-9.0/2.0 * np.log(sigma)**2)
 
  return qv, qc, qt, qs, rw, rm, nc