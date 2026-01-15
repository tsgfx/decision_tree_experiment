import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

try:
    from utils import ensure_dirs
except ImportError:
    def ensure_dirs(base_dir):
        figures_dir = os.path.join(base_dir, "results", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        return None, figures_dir

RANDOM_STATE = 42


def run_joint_constraint_exp(X, y):
    """
    执行深度(1-10) 与 宽度(2,4,8,16,32,64) 的联合约束实验
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    depths = range(1, 11)  # max_depth: 1 to 10
    leaf_nodes_list = [2, 4, 8, 16, 32, 64]  # max_leaf_nodes

    results = []

    for d in depths:
        for l in leaf_nodes_list:
            clf = DecisionTreeClassifier(
                max_depth=d,
                max_leaf_nodes=l,
                random_state=RANDOM_STATE
            )
            clf.fit(X_tr, y_tr)
            pred_te = clf.predict(X_te)
            acc = accuracy_score(y_te, pred_te)

            results.append({
                "max_depth": d,
                "max_leaf_nodes": l,
                "test_acc": acc
            })

    return pd.DataFrame(results)


def plot_heatmap(df, figures_dir):
    """
    绘制热图
    """
    # 将数据转换为矩阵形式 (Pivot Table)
    # 行: max_depth, 列: max_leaf_nodes
    pivot_table = df.pivot(index="max_depth", columns="max_leaf_nodes", values="test_acc")

    # 准备绘图
    plt.figure(figsize=(8, 6), dpi=200)

    # 使用 imshow 绘制热图
    # origin='lower' 让 max_depth=1 在最下方
    im = plt.imshow(pivot_table.values, cmap='viridis', origin='lower', aspect='auto')

    # 设置坐标轴标签
    # X轴: max_leaf_nodes
    plt.xlabel("max_leaf_nodes")
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)

    # Y轴: max_depth
    plt.ylabel("max_depth")
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)

    plt.title("Breast Cancer: Depth-Width Joint Constraint (Test Acc)")

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label("Test Accuracy")

    # 在每个格子里显示数值
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            # 根据背景色深浅调整文字颜色
            text_color = "white" if val < 0.925 else "black"
            plt.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color=text_color, fontsize=8)

    plt.tight_layout()

    # 保存
    filename = "BreastCancer_joint_heatmap.png"
    save_path = os.path.join(figures_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"热图已保存: {filename}")


def main():
    print("=== 正在生成深度-宽度联合约束热图 (Breast Cancer) ===")

    # 1. 准备数据
    X, y = load_breast_cancer(return_X_y=True)

    # 2. 准备路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _, figures_dir = ensure_dirs(base_dir)

    # 3. 运行实验
    df = run_joint_constraint_exp(X, y)

    # 4. 绘图
    plot_heatmap(df, figures_dir)

    print("=== 完成 ===")


if __name__ == "__main__":
    main()