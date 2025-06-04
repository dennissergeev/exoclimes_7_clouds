'''
  We use a simple Euler time-stepping scheme for this example - can be greatly improved using

'''

import numpy as np
from atm_module import q_s_sat

def tracer_sat_adj(nlay, t_step, vap_VMR, vap_mw, cld_sp, rho_d, cld_mw, Tl, pl, rho, met, tau_cond, q_v, q_c):

  # Find the saturation vapour mass mixing ratio at each layer
  q_s = np.zeros(nlay)
  for k in range(nlay):
    q_s[k] = q_s_sat(vap_mw, cld_sp, Tl[k], pl[k], rho[k], met)

  # Usually a suitable first guess of a sub-stepping timescale is the condensation timescale
  t_sub = tau_cond

  # Perform sub-time-stepping integration
  t_now = 0.0
  while t_now < t_step:

    # Avoid overshooting the timestep value
    if ((t_now + t_sub) > t_step):
      t_sub = t_step - t_now

    for k in range(nlay):

      if (q_v[k] < q_s[k]):
        # Vapour is undersaturated - adjust vapour by evaporating from q_c
        dqdt_v = np.minimum(q_s[k] - q_v[k], q_c[k])/tau_cond
      elif (q_v[k] > q_s[k]):
        # Vapour is supersaturated - adjust vapour by condensing from q_v
        dqdt_v = -(q_v[k] - q_s[k])/tau_cond
      else:
        # No nothing as q_v = q_s
        dqdt_v = 0.0
        continue

      # The condensate rate of change is just the negative of the vapour
      dqdt_c = -dqdt_v

      # Perform the Euler time-stepping step for each tracer
      q_v[k] = q_v[k] + t_sub * dqdt_v
      q_c[k] = q_c[k] + t_sub * dqdt_c


    # Add the sub-timestep to the current time
    t_now += t_sub

  return q_v, q_c, q_s