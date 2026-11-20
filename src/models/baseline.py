"""Classical baseline: hand-crafted features + Random Forest (or SVM)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_sklearn(name: str, seed: int = 0, **params):
    if name == "rf":
        clf = RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    elif name == "svm":
        clf = SVC(probability=True, random_state=seed, **params)
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), clf)
