import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42


def ensure_dirs(base_dir):
    """确保结果目录存在"""
    tables_dir = os.path.join(base_dir, "results", "tables")
    figures_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return tables_dir, figures_dir


def tree_max_width(clf):
    """计算树的最大层宽"""
    tree = clf.tree_
    q = deque([(0, 0)])
    level_counts = defaultdict(int)
    while q:
        nid, depth = q.popleft()
        level_counts[depth] += 1
        l, r = tree.children_left[nid], tree.children_right[nid]
        if l != -1:
            q.append((l, depth + 1))
        if r != -1:
            q.append((r, depth + 1))
    return max(level_counts.values()) if level_counts else 0


def run_depth_sweep(X, y, depths=range(1, 21)):
    """执行深度扫掠实验"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    rows = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        pred_tr = clf.predict(X_tr)
        pred_te = clf.predict(X_te)
        rows.append({
            "max_depth": d,
            "actual_depth": clf.tree_.max_depth,
            "node_count": clf.tree_.node_count,
            "n_leaves": clf.get_n_leaves(),
            "max_width": tree_max_width(clf),
            "train_acc": accuracy_score(y_tr, pred_tr),
            "test_acc": accuracy_score(y_te, pred_te),
            "train_f1": f1_score(y_tr, pred_tr, average="macro"),
            "test_f1": f1_score(y_te, pred_te, average="macro"),
        })
    return pd.DataFrame(rows)


def run_leaf_sweep(X, y, leaf_nodes=(2, 4, 8, 16, 32, 64, 128, 256)):
    """执行宽度(叶节点)扫掠实验"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    rows = []
    for L in leaf_nodes:
        clf = DecisionTreeClassifier(max_leaf_nodes=L, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        pred_tr = clf.predict(X_tr)
        pred_te = clf.predict(X_te)
        rows.append({
            "max_leaf_nodes": L,
            "actual_depth": clf.tree_.max_depth,
            "node_count": clf.tree_.node_count,
            "n_leaves": clf.get_n_leaves(),
            "max_width": tree_max_width(clf),
            "train_acc": accuracy_score(y_tr, pred_tr),
            "test_acc": accuracy_score(y_te, pred_te),
            "train_f1": f1_score(y_tr, pred_tr, average="macro"),
            "test_f1": f1_score(y_te, pred_te, average="macro"),
        })
    return pd.DataFrame(rows)


def cv_accuracy(X, y, params_list, n_splits=5, n_repeats=3):
    """执行重复分层交叉验证"""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    out = []
    for params in params_list:
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        scores = cross_val_score(clf, X, y, cv=rskf, scoring="accuracy")
        out.append({**params, "cv_mean": scores.mean(), "cv_std": scores.std(ddof=1)})
    return pd.DataFrame(out)


def plot_and_save(df_main, df_cv, x_col, dataset_name, figures_dir, mode="depth"):
    """绘图通用函数"""
    plt.figure(figsize=(6.5, 4.2), dpi=200)
    plt.plot(df_main[x_col], df_main["train_acc"], label="train")
    plt.plot(df_main[x_col], df_main["test_acc"], label="test")
    plt.errorbar(df_cv[x_col], df_cv["cv_mean"], yerr=df_cv["cv_std"],
                 linestyle="--", capsize=2, label="CV mean±1σ")

    if mode == "leaf":
        plt.xscale("log", base=2)

    plt.xlabel(x_col)
    plt.ylabel("accuracy")
    plt.title(f"{dataset_name}: {mode} sweep")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()

    filename = f"{dataset_name}_{mode}.png"
    plt.savefig(os.path.join(figures_dir, filename), bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {filename}")