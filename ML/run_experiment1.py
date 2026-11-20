import os
from models import train
import time
import dataset
import utils
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs.local")
args = parser.parse_args()

config = importlib.import_module(args.config)

print("=" * 60)
print(f"Yield prediction LOPO-CV crop: {config.CROP}")
print("=" * 60)

df, target, feature_cols = dataset.load_data_crop(config)
experiment_dir = os.path.join(config.WD, config.RESULTS_FOLDER, config.CROP)
os.makedirs(experiment_dir, exist_ok=True)
for model_name in config.models.keys():
    model_dict = config.models[model_name]
    model =  model_dict['model']
    param_grid = model_dict['param_grid']
    for year in target.index.unique():
        print("=" * 60, f"year {year}", "=" * 60)
        out_dir = os.path.join(experiment_dir, str(year))
        os.makedirs(out_dir, exist_ok=True)
        year_data = df.loc[year]
        year_target = target.loc[year]
        year_data_reduced = dataset.check_nan(year_data)
        feature_cols_reduced = [col for col in feature_cols if col in year_data_reduced.columns]
        utils.json_dump(out_dir, f'features_{year}', feature_cols_reduced)
        dataset.neg_soil(year_data_reduced)
        start = time.time()
        train.lopo_cv(year_data_reduced, year_target, feature_cols_reduced, model, param_grid,
                      config.PARCEL_COL, config.YEAR_COL,
                      out_dir, model_name, config.N_ITER_SEARCH)
        end = time.time()
        execution_time = end - start
        utils.txt_dump(out_dir, f'exp_time_{config.CROP}_{year}_{model_name}', str(execution_time))
