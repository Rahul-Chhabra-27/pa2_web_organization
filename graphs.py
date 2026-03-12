import numpy as np
import matplotlib.pyplot as plt
z = np.load("B2_mul_gru_noclip_final_state.npz")
grad_time = z["grad_time"] # (num_checkpoints, Tstore), NaN padded
sat_time = z["sat_time"] # (num_checkpoints, Tstore)
valid_err = z["valid_error"] # (num_checkpoints,)
rho = z["rho_Whh"] # (num_checkpoints,)
# choose a checkpoint (e.g., last non-empty)
g = grad_time[-1]; s = sat_time[-1]
g = g[np.isfinite(g)]; s = s[np.isfinite(s)]
plt.figure(); plt.hist(np.log10(g + 1e-12), bins=60); plt.title("log10␣||dL/dh_t||")
plt.figure(); plt.hist(s, bins=60, range=(0,1)); plt.title("hidden␣saturation␣distance")
plt.figure(); plt.plot(valid_err); plt.title("validation␣error␣(%)")
plt.show()