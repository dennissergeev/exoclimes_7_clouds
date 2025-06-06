# Exoclimes VII 3D clouds practical
 
Python based practical for Exoclimes VII cloud modelling review talk
This repository acts as a supplemental to the presentation (included here too).

NOTE: The best way to generate a non-irradiated profile (e.g. brown dwarf) is set the zenith angle to a small value (e.g. mu_z = 1e-6), rather than set Tirr = 0. This is just due to the (probably bad) way the semi-grey and picket fence T-p profile calculation was coded.


## Key papers for different methods (non-exhaustive)

### Tsuji

### Rossow

### Ackerman & Marley

### Tracer saturation adjustment

### DRIFT

### Mass moment method (mini-cloud)

### CARMA


## DISCLAIMER:

These are codes for learning basics and toy modelling only, and are NOT complete with regards to physics/chemistry in-depth details (which matter a lot), may contain bugs, and are not as tested as the original implementations. They are also not numerically optimised (on purpose for readability).
So do not use these for end product science, consult an expert in the field for example (non-exhaustive list) A&M -> VIRGA [Natasha Batalha, Caroline Morley, Mark Marley], tracer sat adj -> [Elspeth Lee, Xianyu Tan, Tad Komacek], two mass moment microphysics -> [Elspeth Lee, Kazumasa Ohno] if you want to go further with any of the methods for `real' science.

For bin models (not used here) consult a CARMA expert e.g. Diana Powell, Peter Gao. \\
For interest in the DRIFT moment methodology consult Christiane Helling or Peter Woitke. \\
For interest in advanced size distribution dependent mass moment methods consult Elspeth Lee or Kazumasa Ohno. \\
