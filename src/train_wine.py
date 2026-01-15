import os
from sklearn.datasets import load_wine
from utils import ensure_dirs, run_depth_sweep, run_leaf_sweep, cv_accuracy, plot_and_save


def main():
    # 1. 准备数据
    name = "Wine"
    print(f"=== 正在运行 {name} 实验 ===")
    X, y = load_wine(return_X_y=True)

    # 2. 准备路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tables_dir, figures_dir = ensure_dirs(base_dir)

    # 3. 运行深度扫掠
    print("运行深度扫掠...")
    df_depth = run_depth_sweep(X, y)
    depth_settings = [{"max_depth": d} for d in range(1, 21)]
    cv_depth = cv_accuracy(X, y, depth_settings)

    df_depth.to_csv(os.path.join(tables_dir, f"{name}_depth.csv"), index=False)
    cv_depth.to_csv(os.path.join(tables_dir, f"{name}_cv_depth.csv"), index=False)
    plot_and_save(df_depth, cv_depth, "max_depth", name, figures_dir, mode="depth")

    # 4. 运行宽度扫掠
    print("运行宽度扫掠...")
    df_leaf = run_leaf_sweep(X, y)
    leaf_settings = [{"max_leaf_nodes": L} for L in (2, 4, 8, 16, 32, 64, 128, 256)]
    cv_leaf = cv_accuracy(X, y, leaf_settings)

    df_leaf.to_csv(os.path.join(tables_dir, f"{name}_leaf.csv"), index=False)
    cv_leaf.to_csv(os.path.join(tables_dir, f"{name}_cv_leaf.csv"), index=False)
    plot_and_save(df_leaf, cv_leaf, "max_leaf_nodes", name, figures_dir, mode="leaf")

    print(f"=== {name} 实验完成 ===")


if __name__ == "__main__":
    main()