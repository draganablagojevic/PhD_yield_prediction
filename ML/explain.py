import os
import numpy as np
import shap
import pandas as pd
import utils
import matplotlib.pyplot as plt

def make_shap_explainer(model, save_filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"shap_explainer_{save_filename}"
    explainer_path = os.path.join(output_dir, f"{file_name}.pkl")

    if os.path.exists(explainer_path):
        print(f"Loading existing SHAP explainer from {output_dir}")
        shap_expl_object = utils.pickle_load(output_dir, f"{file_name}")
    else:
        print(f"Calculating SHAP explainer for {save_filename}")
        shap_expl_object = shap.TreeExplainer(model)
        utils.pickle_dump(output_dir, file_name, shap_expl_object)
    return shap_expl_object

def calculate_mdi(model, output_dir, save_filename, feature_names=None):
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"mdi_{save_filename}"
    explainer_path = os.path.join(output_dir, f"{file_name}.pkl")
    if os.path.exists(explainer_path):
        print(f"Loading existing MDI explainer from: {output_dir}")
        importance= utils.pickle_load(output_dir, file_name)
    else:
        print(f"Calculating MDI for {save_filename}")
        importance = np.asarray(model.feature_importances_)
        utils.pickle_dump(output_dir, file_name, importance)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
    else:
        if len(feature_names) != len(importance):
            raise ValueError("Number of feature names must match number of features.")
    mdi = pd.DataFrame({"feature": feature_names,"mdi": importance})
    return mdi.sort_values("mdi", ascending=False).reset_index(drop=True)

def shap_feature_importance(shap_expl_object, X, feature_names,
                            output_path, save_file_name, title = "Feature Importance", color = 'blue', top_n=10):
    shap_values = shap_expl_object.shap_values(X)
    # Mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # Keep only features with non-zero SHAP importance
    nonzero_indices = np.where(mean_abs_shap > 0)[0]
    n_features = min(len(nonzero_indices), top_n)
    top_indices = nonzero_indices[np.argsort(mean_abs_shap[nonzero_indices])[-n_features:][::-1]]
    shap_values_selected = shap_values[:, top_indices]
    X_selected = X.iloc[:, top_indices]
    feature_names_selected = np.array(feature_names)[top_indices]
    plt.figure()
    shap.summary_plot(shap_values_selected,X_selected,feature_names=feature_names_selected,plot_type="bar",show=False)
    ax = plt.gca()

    for bar in ax.patches:
        bar.set_facecolor(color)
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(output_path, save_file_name)
    plt.savefig(save_path, bbox_inches="tight", dpi = 300)
    plt.close()

def mdi_feature_importance(mdi, output_path, save_file_name, top_n=10, title="Mean decrease in impurity (MDI)",
             color="blue"):
    os.makedirs(output_path, exist_ok=True)
    if not {"feature", "mdi"}.issubset(mdi.columns):
        raise ValueError("mdi must contain 'feature' and 'mdi' columns.")

    data = mdi.sort_values("mdi", ascending=True)

    if top_n is not None:
        data = data.tail(top_n)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.barh(data["feature"], data["mdi"], color=color)

    ax.set_xlabel("MDI")
    ax.set_ylabel("Feature names")
    ax.set_title(title)

    for i, value in enumerate(data["mdi"]):
        ax.text(value,i,f" {value:.3f}",va="center")
    plt.tight_layout()
    save_path = os.path.join(output_path, save_file_name)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


