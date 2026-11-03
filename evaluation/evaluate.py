import pickle, numpy as np, json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
rows = pickle.load(open("features.pkl", "rb"))
G = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); K = np.array([r[2] for r in rows])
names = list(rows[0][3]); X = np.array([[r[3][n] for n in names] for r in rows]); W = np.array([r[4] for r in rows])
def acc(t, p): return float((t == p).mean())
def report(tag, t, p, kinds):
    tp = ((p==1)&(t==1)).sum(); tn=((p==0)&(t==0)).sum(); fp=((p==1)&(t==0)).sum(); fn=((p==0)&(t==1)).sum()
    bal = (tp/(tp+fn) + tn/(tn+fp))/2
    per = {k: round(float((p[kinds==k]==1).mean()),3) for k in ["upsample","grid","smooth","splice"]}
    print(f"{tag:46s} acc={acc(t,p):.3f} balanced={bal:.3f} real-correct={tn/(tn+fp):.3f} fake-caught={tp/(tp+fn):.3f} per-fake-recall={per}")
    return dict(acc=acc(t,p), balanced=float(bal), real_correct=float(tn/(tn+fp)), fake_recall=float(tp/(tp+fn)), per_fake=per)
res = {}
print(f"n={len(y)} (real={int((y==0).sum())}, fake={int((y==1).sum())}), 14 source photos")
res["shipped_rules"] = report("1. shipped rule flags (>=1 warn => fake)", y, (W>=1).astype(int), K)
def cv(make, tag, split="source"):
    p = np.zeros(len(y), int)
    for tr, te in LeaveOneGroupOut().split(X, y, G):
        m = make().fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return report(tag, y, p, K)
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
gb = lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
res["logreg_loso"] = cv(lr, "2. logistic reg, leave-one-PHOTO-out")
res["gboost_loso"] = cv(gb, "3. grad-boost, leave-one-PHOTO-out")
# unseen fake family: train without family f, test on real(held-out photos) + family f
print("--- unseen fake-family generalisation (train excludes that family; photos also held out)")
unseen = {}
for fam in ["upsample","grid","smooth","splice"]:
    p = np.zeros(len(y), int); mask = np.zeros(len(y), bool)
    for tr, te in LeaveOneGroupOut().split(X, y, G):
        tr = tr[K[tr] != fam]; te = te[(K[te]=="real") | (K[te]==fam)]
        m = gb().fit(X[tr], y[tr]); p[te] = m.predict(X[te]); mask[te] = True
    t = y[mask]; pp = p[mask]
    unseen[fam] = dict(acc=acc(t,pp), fake_recall=float((pp[t==1]==1).mean()), real_correct=float((pp[t==0]==0).mean()))
    print(f"   unseen={fam:9s} acc={unseen[fam]['acc']:.3f} fake-recall={unseen[fam]['fake_recall']:.3f} real-correct={unseen[fam]['real_correct']:.3f}")
res["unseen_family_gboost"] = unseen
json.dump(res, open("results.json","w"), indent=1)
