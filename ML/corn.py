from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

WD = r'/mnt/netapp/home/blagojevicd/PhD'
DATA_FOLDER = 'Final_dataset_v3'
DATASET_FILE_NAME = r"ml_pixel_dataset.csv"
RESULTS_FOLDER = r"Exp1_dv3_results"
CROP = "corn"
TARGET_COL = "yield"
PARCEL_COL = "parcel"
YEAR_COL = "year"

DROP_COLS = ["crop", "Latitude", "Longitude", "SOS_date", "POS_date", "EOS_date", "sowing_date", "harvest_date",
    
	"pixel_id", "geometry", "point", "Latitude", "Longitude",
    	"sowing_date", "harvest_date", "season_days",
    	"moisture", "speed", "crop"]

N_ITER_SEARCH = 20

DT_PARAM_GRID = {'criterion': ["squared_error", "absolute_error", "poisson"],
                 'max_depth': [2, 3, 4, 5, 10, 15, 20],
                 'min_samples_leaf':[5, 10, 20, 50]}

RF_PARAM_GRID = {'max_depth': [5, 10, 15, 20],
                 'min_samples_leaf': [5, 10, 20, 50],
                 'n_estimators':     [5, 10, 50, 100]}

XGB_PARAM_GRID = {
    'n_estimators':     [5, 10, 50, 100],
    'learning_rate':    [0.05, 0.1, 0.2],
    'max_depth':        [3, 5, 7],
    'min_child_weight': [10, 20, 50],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

models = {'DT': {'model': DecisionTreeRegressor(),'param_grid': DT_PARAM_GRID},
          'RF':  {'model': RandomForestRegressor(), 'param_grid': RF_PARAM_GRID},
          'XGBoost': {'model': XGBRegressor(), 'param_grid': XGB_PARAM_GRID}}