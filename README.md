# Crop yield prediction
This is the pipeline and set of ML experiments I use to predict field-scale crop yields in Vojvodina, Serbia. It covers wheat, corn, sunflower and soybeans over four seasons (2017-2021), at 10 m pixel level and parcel level.

Data: yield monitor data from combine harvesters, cleaned and placed on a 10 m grid matched to Sentinel-2. It is combined with Sentinel-2 and Landsat 7/8 imagery (with my own Landsat→Sentinel-2 harmonization), 15 vegetation indices, phenology metrics, ERA5-Land climate summarized by crop growth stage (BBCH), terrain derivatives and SoilGrids soil properties.
Dataset: about 713k pixels, 106 parcel-seasons and 287 predictors.
Models: Decision Tree, Random Forest and XGBoost.
Validation: nested cross-validation grouped by parcel (leave-one-parcel-out) and an expanding window over years (training on past seasons, testing on a later one). This keeps spatial and temporal autocorrelation from leaking between training and test data.
Stack: Python (pandas, geopandas, rasterio, scikit-learn, XGBoost) on an HPC cluster with SLURM.
