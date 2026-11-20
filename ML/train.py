from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import os
import pandas as pd
import utils
from evaluate import evaluate_model
from models.explain import make_shap_explainer, calculate_mdi, shap_feature_importance, mdi_feature_importance

def expanding_year_cv(data, year_column, min_train_years=1):
    years = np.sort(data[year_column].unique())
    splits = []
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_idx = np.where(data[year_column].isin(train_years).to_numpy())[0]
        test_idx = np.where((data[year_column] == test_year).to_numpy())[0]
        splits.append((train_idx, test_idx))
    return splits

def cv_iterator(data, target, group_by_column):
    """LOPO CV iterator"""
    logo = LeaveOneGroupOut()
    groups = data[group_by_column]
    splits = list(logo.split(data, target, groups=groups))
    return splits


def lopo_cv(df, target, feature_cols, model, param_grid,
            parcel_col_name, year_col,
            out_dir, model_name, n_iter):

    parcel_list = sorted(df[parcel_col_name].unique())
    pred_all_parcels  = []
    metrics_list   = []
    best_params_dict = {}

    for idx, parcel in enumerate(parcel_list):
        print(f"    [{idx+1}/{len(parcel_list)}] Test parcela: {parcel}")

        test_mask = df[parcel_col_name] == parcel
        X_test = df[test_mask]
        X_train = df[~test_mask]

        target_test = target[test_mask]
        target_train = target[~test_mask]

        cv_splits = cv_iterator(X_train, target_train, f'{parcel_col_name}')

        X_test = X_test[feature_cols]
        X_train = X_train[feature_cols]

        search = RandomizedSearchCV(estimator=model,
                                    param_distributions=param_grid,
                                    n_iter=n_iter,
                                    cv=cv_splits,
                                    scoring="r2",
                                    n_jobs=-1,
                                    verbose=1,
                                    random_state=42,
                                    error_score=np.nan)

        search.fit(X_train, target_train)
        best_params_dict[parcel] = search.best_params_
        print(f"Best params: {search.best_params_}")
        utils.pickle_dump(out_dir, f"{model_name}_params_{parcel}", search.best_params_)

        if hasattr(search, 'cv_results_'):
            cv_df = pd.DataFrame(search.cv_results_)[["mean_fit_time",
                                                      "std_fit_time",
                                                      "mean_score_time",
                                                      "std_score_time",
                                                      "param_max_depth",
                                                      "params",
                                                      "mean_test_score",
                                                      "std_test_score",
                                                      "rank_test_score"]]
            cv_df["test_parcel"] = parcel
            cv_df.to_excel(os.path.join(out_dir, f"{model_name}_search_{parcel}.xlsx"), index=False)

        best_model = search.best_estimator_
        utils.pickle_dump(out_dir, f"{model_name}_parcel_{parcel}_model", best_model)
        y_pred = best_model.predict(X_test)
        shap_dir = os.path.join(out_dir, f"shap_{model_name}")
        shap_expl_object = make_shap_explainer(best_model, f"{model_name}_{parcel}", shap_dir)
        shap_feature_importance(shap_expl_object, X_test, feature_cols, shap_dir,
                                f"{parcel}_feature_importance", title = f"{parcel} feature importance")
        mdi_dir = os.path.join(out_dir, f"mdi_{model_name}")
        mdi = calculate_mdi(best_model, mdi_dir, f"{model_name}_{parcel}", feature_cols)
        mdi_feature_importance(mdi, mdi_dir, f"{parcel}_feature_importance_mdi",
                               top_n=10, title="Mean decrease in impurity (MDI)",
                               color="blue")

        metrics = evaluate_model(target_test.values, y_pred)
        metrics["test_parcel"] = parcel
        metrics["n_test"] = len(target_test)
        metrics["n_train"] = len(target_train)
        metrics["best_params"] = str(search.best_params_)
        metrics["year"] = str(sorted(target_test.index.unique()))
        metrics_list.append(metrics)

        pred_df = pd.DataFrame({
            parcel_col_name: parcel,
            year_col:   target_test.index,
            "y_test":   target_test.values,
            "y_pred":   y_pred,
        })
        #pred_df.to_csv(os.path.join(out_dir,f"{model_name}_pred_{parcel}.csv"), index=False)
        pred_all_parcels.append(pred_df)

    merged_pred_all_parcels = pd.concat(pred_all_parcels, ignore_index=True)
    merged_pred_all_parcels.to_csv(os.path.join(out_dir, f'{model_name}_all_predictions.csv'), index = False)
    metrics_all_parcels = evaluate_model(
        merged_pred_all_parcels["y_test"].values, merged_pred_all_parcels["y_pred"].values)
    metrics_all_parcels[parcel_col_name]  = "ALL"
    metrics_df = pd.DataFrame(metrics_list + [metrics_all_parcels])
    metrics_df.to_csv(os.path.join(out_dir, f"{model_name}_all_parcels_results.csv"), index = False)
    utils.pickle_dump(out_dir, f"{model_name}_all_params", best_params_dict)

def loyo_cv(df, target, feature_cols, model, param_grid, year_col, out_dir, model_name, n_iter):
    shap_dir = os.path.join(out_dir, f"shap_{model_name}")
    year_list = sorted(int(year) for year in df[year_col].unique())
    pred_all_years  = []
    metrics_list   = []
    best_params_dict = {}
    outer_splits = expanding_year_cv(df, year_column=year_col, min_train_years=2)

    for i, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        train_years = year_list[:i]
        test_year = year_list[i]
        print(f"[{i+1}/{len(year_list)}] Test year: {test_year}, Train years: {train_years}")

        X_test = df.iloc[outer_test_idx]
        X_train = df.iloc[outer_train_idx]

        target_test = target.iloc[outer_test_idx]
        target_train = target.iloc[outer_train_idx]

        cv_splits = expanding_year_cv(X_train,year_column=year_col,min_train_years=1)

        X_test = X_test[feature_cols]
        X_train = X_train[feature_cols]

        search = RandomizedSearchCV(estimator=model,
                                    param_distributions=param_grid,
                                    n_iter=n_iter,
                                    cv=cv_splits,
                                    scoring="r2",
                                    n_jobs=-1,
                                    verbose=1,
                                    random_state=42,
                                    error_score=np.nan)
        print(target_train.head())
        search.fit(X_train, target_train)
        best_params_dict[test_year] = search.best_params_
        print(f"Best params: {search.best_params_}")
        utils.pickle_dump(out_dir, f"{model_name}_params_{test_year}", search.best_params_)

        if hasattr(search, 'cv_results_'):
            cv_df = pd.DataFrame(search.cv_results_)[["mean_fit_time",
                                                      "std_fit_time",
                                                      "mean_score_time",
                                                      "std_score_time",
                                                      "param_max_depth",
                                                      "params",
                                                      "mean_test_score",
                                                      "std_test_score",
                                                      "rank_test_score"]]
            cv_df["test_year"] = test_year
            cv_df.to_excel(os.path.join(out_dir, f"{model_name}_search_{test_year}.xlsx"), index=False)

        best_model = search.best_estimator_
        utils.pickle_dump(out_dir, f"{model_name}_parcel_{test_year}", best_model)
        y_pred = best_model.predict(X_test)

        shap_expl_object = make_shap_explainer(best_model, f"{model_name}_{test_year}", shap_dir)
        shap_feature_importance(shap_expl_object, X_test, feature_cols, shap_dir,
                                f"{test_year}_feature_importance", title=f"{test_year} feature importance")
        mdi_dir = os.path.join(out_dir, f"mdi_{model_name}")
        mdi = calculate_mdi(best_model, mdi_dir, f"{model_name}_{test_year}", feature_cols)
        mdi_feature_importance(mdi, mdi_dir, f"{test_year}_feature_importance_mdi",
                               top_n=10, title="Mean decrease in impurity (MDI)",
                               color="blue")
        metrics = evaluate_model(target_test.values, y_pred)
        metrics["test_year"] = test_year
        metrics["n_test"] = len(target_test)
        metrics["n_train"] = len(target_train)
        metrics["best_params"] = str(search.best_params_)
        metrics_list.append(metrics)

        pred_df = pd.DataFrame({'parcel': target_test.index,
            year_col:   test_year,
            "y_test":   np.asarray(target_test, dtype=float).reshape(-1),
            "y_pred":   y_pred,
        })
        #pred_df.to_csv(os.path.join(out_dir,f"{model_name}_pred_{test_year}.csv"), index=False)
        pred_all_years.append(pred_df)

    merged_pred_all_parcels = pd.concat(pred_all_years, ignore_index=True)
    merged_pred_all_parcels.to_csv(os.path.join(out_dir, f'{model_name}_all_predictions.csv'))
    metrics_all_parcels = evaluate_model(
        merged_pred_all_parcels["y_test"].values, merged_pred_all_parcels["y_pred"].values)
    metrics_all_parcels[year_col]  = "ALL"
    metrics_df = pd.DataFrame(metrics_list + [metrics_all_parcels])
    metrics_df.to_csv(os.path.join(out_dir, f"{model_name}_all_parcels_results.csv"), index = False)
    utils.pickle_dump(out_dir, f"{model_name}_all_params", best_params_dict)