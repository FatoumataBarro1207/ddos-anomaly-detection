#!/usr/bin/env python3
"""
Mini-projet : détection d'anomalies (DDoS) sur des flux réseau.

Deux approches comparées sur le même jeu de test :
  1. Isolation Forest entraînée UNIQUEMENT sur du trafic normal (détection d'anomalies,
     sans utiliser les étiquettes d'attaque pour apprendre).
  2. Random Forest supervisée (référence de comparaison).

Données visées : un fichier CSV de CIC-IDS2017 (journée du vendredi, DDoS), mais le script
accepte tout CSV de flux avec une colonne « Label » dont la valeur normale est « BENIGN ».

Utilisation :
    python ddos_anomaly.py --demo                     # vérifier l'installation (données synthétiques)
    python ddos_anomaly.py --csv chemin/vers/fichier.csv --out results

ATTENTION : les résultats du mode --demo n'ont aucune valeur scientifique. Ne les citez jamais.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42

# Colonnes d'identification / « raccourcis » à retirer : un modèle qui s'appuie sur une adresse IP,
# un port ou une date n'apprend pas le comportement d'une attaque, il apprend le scénario de capture.
SHORTCUT_COLS = {
    "flow id", "source ip", "destination ip", "src ip", "dst ip", "timestamp",
    "source port", "destination port", "src port", "dst port",
    "fwd header length.1",  # colonne dupliquée présente dans CIC-IDS2017
}


# ----------------------------------------------------------------------------- données
def make_demo_data(n=20000, n_features=30, attack_ratio=0.3, seed=SEED):
    """Données SYNTHÉTIQUES, uniquement pour vérifier que le pipeline s'exécute."""
    rng = np.random.default_rng(seed)
    n_att = int(n * attack_ratio)
    n_ben = n - n_att
    cols = [f"feature_{i}" for i in range(n_features)]
    benign = rng.lognormal(mean=1.0, sigma=1.0, size=(n_ben, n_features))
    attack = rng.lognormal(mean=1.8, sigma=0.8, size=(n_att, n_features))
    X = np.vstack([benign, attack])
    df = pd.DataFrame(X, columns=cols)
    df["Label"] = ["BENIGN"] * n_ben + ["DDoS"] * n_att
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_and_clean(df):
    """Nettoyage : noms de colonnes, valeurs infinies/manquantes, doublons, colonnes raccourcis."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        raise ValueError("Colonne « Label » introuvable. Colonnes vues : " + ", ".join(df.columns[:10]))
    labels = df[label_col].astype(str).str.strip()
    df = df.drop(columns=[label_col])

    drop = [c for c in df.columns if c.lower() in SHORTCUT_COLS]
    df = df.drop(columns=drop)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    n0 = len(df)
    df["__label__"] = labels.values
    df = df.dropna()
    n_nan = n0 - len(df)
    df = df.drop_duplicates()
    n_dup = n0 - n_nan - len(df)

    y = (df["__label__"].str.upper() != "BENIGN").astype(int).values
    label_names = df["__label__"].values
    X = df.drop(columns=["__label__"])
    const = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const)

    info = {
        "lignes_initiales": int(n0),
        "lignes_supprimees_valeurs_manquantes_ou_infinies": int(n_nan),
        "lignes_supprimees_doublons": int(n_dup),
        "lignes_finales": int(len(X)),
        "colonnes_raccourcis_retirees": drop,
        "colonnes_constantes_retirees": const,
        "n_variables": int(X.shape[1]),
        "part_attaque": float(y.mean()),
        "etiquettes_presentes": pd.Series(label_names).value_counts().to_dict(),
    }
    return X, y, info


# ----------------------------------------------------------------------------- évaluation
def metrics_at_threshold(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(p), "rappel": float(r), "f1": float(f1),
        "taux_fausses_alertes": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn),
    }


def run(X, y, out_dir, sample=None, test_attack_ratio=None, dropped=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample and sample < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=sample, stratify=y, random_state=SEED)
        X = X.reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    # Option : jeu de test « réaliste », où les attaques sont rares (l'entraînement ne change pas).
    if test_attack_ratio is not None:
        rng = np.random.default_rng(SEED)
        ben_idx = np.where(y_te == 0)[0]
        att_idx = np.where(y_te == 1)[0]
        n_att = int(round(test_attack_ratio / (1 - test_attack_ratio) * len(ben_idx)))
        n_att = max(1, min(n_att, len(att_idx)))
        keep = np.sort(np.concatenate([ben_idx, rng.choice(att_idx, n_att, replace=False)]))
        X_te, y_te = X_te.iloc[keep], y_te[keep]

    # --- 1. Isolation Forest : apprentissage sur du trafic normal uniquement
    X_ben = X_tr[y_tr == 0]
    X_fit, X_val = train_test_split(X_ben, test_size=0.2, random_state=SEED)
    iforest = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_fit)
    score_val = -iforest.score_samples(X_val)      # plus grand = plus anormal
    score_te = -iforest.score_samples(X_te)
    thr = float(np.quantile(score_val, 0.99))      # seuil : ~1 % de fausses alertes sur du trafic normal
    if_pred = (score_te > thr).astype(int)

    res_if = metrics_at_threshold(y_te, if_pred)
    res_if["PR_AUC"] = float(average_precision_score(y_te, score_te))
    res_if["ROC_AUC"] = float(roc_auc_score(y_te, score_te))
    res_if["seuil"] = thr

    # --- 2. Random Forest supervisée (référence)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    proba_te = rf.predict_proba(X_te)[:, 1]
    res_rf = metrics_at_threshold(y_te, (proba_te >= 0.5).astype(int))
    res_rf["PR_AUC"] = float(average_precision_score(y_te, proba_te))
    res_rf["ROC_AUC"] = float(roc_auc_score(y_te, proba_te))

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = imp.head(10)

    prevalence = float(y_te.mean())
    results = {
        "configuration": {"test_attack_ratio": test_attack_ratio, "variables_retirees_par_l_utilisateur": dropped or []},
        "jeu_de_test": {"lignes": int(len(y_te)), "part_attaque": prevalence},
        "isolation_forest_(apprise_sur_trafic_normal_seul)": res_if,
        "random_forest_(supervisee)": res_rf,
        "top10_variables_random_forest": {k: float(v) for k, v in top.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- figures
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in (("Isolation Forest", score_te), ("Random Forest", proba_te)):
        p, r, _ = precision_recall_curve(y_te, s)
        ax.plot(r, p, label=name)
    ax.axhline(prevalence, ls="--", color="grey", label=f"hasard ({prevalence:.2f})")
    ax.set_xlabel("Rappel"); ax.set_ylabel("Précision"); ax.set_title("Courbes précision-rappel (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "pr_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    top[::-1].plot.barh(ax=ax)
    ax.set_title("Variables les plus importantes (Random Forest)")
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=150); plt.close(fig)

    # --- tableau lisible
    md = [
        "| Modèle | Précision | Rappel | F1 | Fausses alertes | PR-AUC | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in (("Isolation Forest (normal seul)", res_if), ("Random Forest (supervisée)", res_rf)):
        md.append(f"| {name} | {r['precision']:.3f} | {r['rappel']:.3f} | {r['f1']:.3f} | "
                  f"{r['taux_fausses_alertes']:.3f} | {r['PR_AUC']:.3f} | {r['ROC_AUC']:.3f} |")
    md.append("")
    md.append(f"Part d'attaques dans le jeu de test : {prevalence:.3f} (PR-AUC d'un classifieur au hasard ≈ cette valeur).")
    (out_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")

    return results, "\n".join(md), top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Chemin du fichier CSV de flux (colonne Label requise)")
    ap.add_argument("--demo", action="store_true", help="Données SYNTHÉTIQUES, pour tester l'installation")
    ap.add_argument("--out", default="results", help="Dossier de sortie (défaut : results)")
    ap.add_argument("--sample", type=int, default=None, help="Sous-échantillon stratifié (lignes), pour aller plus vite")
    ap.add_argument("--drop", default=None,
                    help="Variables à retirer, séparées par des virgules (test d'ablation). Ex. : Init_Win_bytes_forward")
    ap.add_argument("--test-attack-ratio", type=float, default=None,
                    help="Part d'attaques voulue dans le jeu de TEST, entre 0 et 1 (ex. 0.02 = 2 %%) ; test plus réaliste")
    a = ap.parse_args()
    if a.test_attack_ratio is not None and not (0 < a.test_attack_ratio < 1):
        ap.error("--test-attack-ratio doit être strictement entre 0 et 1 (ex. 0.02)")

    if a.demo:
        df = make_demo_data()
        print("MODE DÉMO : données synthétiques. Résultats sans valeur scientifique.\n")
    elif a.csv:
        df = pd.read_csv(a.csv, low_memory=False, encoding="utf-8", encoding_errors="replace")
    else:
        ap.error("Indiquez --csv FICHIER ou --demo")

    X, y, info = load_and_clean(df)
    print("Nettoyage :")
    for k, v in info.items():
        print(f"  {k} : {v}")
    if y.min() == y.max():
        raise SystemExit("Une seule classe présente dans les données : impossible d'évaluer.")

    dropped = []
    if a.drop:
        wanted = [c.strip() for c in a.drop.split(",") if c.strip()]
        lookup = {c.lower(): c for c in X.columns}
        dropped = [lookup[w.lower()] for w in wanted if w.lower() in lookup]
        missing = [w for w in wanted if w.lower() not in lookup]
        if missing:
            print("Variables introuvables (ignorées) :", missing)
        X = X.drop(columns=dropped)
        print("Variables retirées pour le test d'ablation :", dropped)

    results, table, top = run(X, y, a.out, a.sample, a.test_attack_ratio, dropped)
    (Path(a.out) / "nettoyage.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRésultats (jeu de test) :\n" + table)
    print("\nVariables les plus importantes (Random Forest) :")
    print(top.round(4).to_string())
    print(f"\nFichiers écrits dans : {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()