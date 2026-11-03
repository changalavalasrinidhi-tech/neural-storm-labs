import pickle, time, numpy as np
from datasets import *
from features2 import full_vector
rng = np.random.default_rng(7)
src = load_real_sources(); names = sorted(src); donors = [src[n] for n in names]; rows = []; t0 = time.time()
for n in names:
    for c in crops(src[n], 36, 256, rng):
        d, w = full_vector(c); rows.append((n, 0, "real", d, w))
        k = FAKE_KINDS[rng.integers(4)]; f = make_fake(k, c, rng, donors)
        d, w = full_vector(f); rows.append((n, 1, k, d, w))
    print(n, len(rows), f"{time.time()-t0:.0f}s", flush=True)
pickle.dump(rows, open("features2.pkl", "wb"))
