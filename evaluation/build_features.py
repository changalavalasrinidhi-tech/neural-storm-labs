import sys, pickle, time, numpy as np
from datasets import *
from features import vector
rng = np.random.default_rng(42)
src = load_real_sources(); names = sorted(src); print(len(names), "real sources:", names)
rows = []  # (group, label, kind, featdict, warn)
t0 = time.time()
donors = [src[n] for n in names]
for gi, n in enumerate(names):
    for c in crops(src[n], 10, 256, rng):
        d, w = vector(c); rows.append((n, 0, "real", d, w))
        for k in FAKE_KINDS:
            f = make_fake(k, c, rng, donors)
            d, w = vector(f); rows.append((n, 1, k, d, w))
    print(n, len(rows), f"{time.time()-t0:.0f}s", flush=True)
pickle.dump(rows, open("features.pkl", "wb"))
