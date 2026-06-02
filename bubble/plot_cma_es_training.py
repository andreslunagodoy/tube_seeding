import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SIGN_FLIP = -1
DIVISOR = 36.0
EMA_SMOOTHING = 0.6
POP_SIZE = 20

# TensorBoard color palette
COLOR_RUN1 = '#4285f4'   # Google Blue 500
COLOR_RUN2 = '#ff7043'   # Deep Orange 400

LOG_FILES = ['run1.log', 'run2.log']
OUTPUT = 'cma_es_training.png'

# ---------------------------------------------------------------------------
def parse_log(path):
    """Parse a CMA-ES log file, returning iteration bests (negated, scaled)."""
    all_costs = []
    with open(path, 'r') as f:
        in_cmaes = False
        for line in f:
            if '(10_w,20)-aCMA-ES' in line or 'CMA-ES' in line:
                in_cmaes = True
                continue
            if 'termination on' in line:
                break
            if in_cmaes:
                m = re.search(r'No\. of arithmetic operations: (\d+)', line)
                if m:
                    all_costs.append(int(m.group(1)) * SIGN_FLIP / DIVISOR)
    iter_bests = []
    for i in range(0, len(all_costs), POP_SIZE):
        chunk = all_costs[i:i + POP_SIZE]
        iter_bests.append(max(chunk))
    return np.array(iter_bests)

# ---------------------------------------------------------------------------
def ema(data, alpha):
    s = np.zeros_like(data)
    s[0] = data[0]
    for i in range(1, len(data)):
        s[i] = alpha * data[i] + (1 - alpha) * s[i - 1]
    return s

# ---------------------------------------------------------------------------
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 5))

for path, color, label in zip(['run2.log', 'run1.log'], [COLOR_RUN2, COLOR_RUN1], ['Rectangular bound', 'Triangular bound']):
    data = parse_log(path)
    smoothed = ema(data, EMA_SMOOTHING)
    iters = np.arange(1, len(data) + 1) * POP_SIZE

    # Faded raw
    ax.plot(iters, data, color=color, linewidth=0.8, alpha=0.3)
    # Solid smoothed
    ax.plot(iters, smoothed, color=color, linewidth=2.0,
            label=label)

    print(f"{path}: {len(data)} iterations, range {data[0]:.1f} → {data[-1]:.1f}, "
          f"best {max(data):.1f}")

ax.set_xlabel('Number of Complete IBP Reductions', fontsize=16.5, color='black')
ax.set_ylabel('Normalized Cost', fontsize=16.5, color='black')
# ax.set_title('CMA-ES Training Cost — best per iteration', fontsize=14,
#             fontweight='normal', color='#333333')
ax.legend(fontsize=15, framealpha=0.9, edgecolor='#cccccc')
ax.grid(True, alpha=0.25, color='#cccccc')
ax.tick_params(colors='black', labelsize=13.5)
ax.ticklabel_format(axis='y', style='plain')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))

fig.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved {OUTPUT}")
plt.close()
