"""Train + honestly score the torch-free forensic classifier on YOUR labelled images.

    python evaluation/train_on_folder.py --real path/to/real_faces --fake path/to/fake_faces [--save models/forensic_head.joblib]

Put real and fake images in two folders (e.g. a Kaggle '140k real and fake faces' subset, FaceForensics++ frames,
DFDC frames). Files sharing a prefix before the first '_' or '-' are treated as the same source/identity and are kept
on the same side of the train/test split, so the reported accuracy is not inflated by near-duplicates.
The printed accuracy is on held-out groups only. Nothing is tuned on it.
"""
import argparse, pathlib, sys, re, numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from features2 import full_vector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load(folder, label, limit, size=256):
    X, y, g, names = [], [], [], None
    files = sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})[:limit]
    for p in files:
        im = cv2.imread(str(p));
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); h, w = im.shape[:2]; s = min(h, w, size)
        im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        d, _ = full_vector(im); names = names or list(d)
        X.append([d[n] for n in names]); y.append(label); g.append(re.split(r"[_-]", p.stem)[0])
    return X, y, g, names

ap = argparse.ArgumentParser(); ap.add_argument("--real", required=True); ap.add_argument("--fake", required=True)
ap.add_argument("--limit", type=int, default=3000); ap.add_argument("--save"); a = ap.parse_args()
Xr, yr, gr, names = load(a.real, 0, a.limit); Xf, yf, gf, _ = load(a.fake, 1, a.limit)
X = np.array(Xr + Xf); y = np.array(yr + yf); g = np.array(["r" + s for s in gr] + ["f" + s for s in gf])
print(f"{len(y)} images ({(y==0).sum()} real / {(y==1).sum()} fake), {len(set(g))} groups")
tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, y, g))
m = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(X[tr], y[tr])
p = m.predict(X[te]); print(classification_report(y[te], p, target_names=["real", "fake"], digits=3))
print("confusion [real,fake]:\n", confusion_matrix(y[te], p)); print("ROC-AUC:", round(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]), 3))
print("HELD-OUT ACCURACY:", round(float((p == y[te]).mean()), 3))
if a.save:
    import joblib; joblib.dump({"model": m.fit(X, y), "features": names}, a.save); print("saved", a.save)
