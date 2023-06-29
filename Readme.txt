This project is composed of 2 main folders: Data and Exploration
Data
-----
This contains:
	- the raw dataset originally provided 
	- augmented datasets comprised of additional augmented and engineered features
	- source data from BLS that was used to augment features


Exploration
-----------
This contains:
	- source code as python notebooks
	- results from various hyper parameter searches
	- model parameters and/or hyper parameters for the final chosen model of each type

The latter two are contained in the various folders with prefix "results_"

The source code is laid out over 4 files:
1. exploration.ipynb comprises of initial exploration of the raw dataset provided as well as the code that augments it with the features sourced from BLS
2. featureEngineering.ipynb transforms and adds new features based on a mutual information (relative to expenses) criterion 
3. xgboost.ipynb applies extreme gradient boosting to predict expense quantiles. Hyperparameters were tuned with hyperopt (bayesian hyper-parameter optimizer). This tuning was run on a compute cluster. The file also includes a superficial feature importance analysis.
4. QuantileMLP.ipynb builds and applies a MLP in Keras to predict expense quantiles. Hyperparameters were tuned with a manual gridcv search run on a compute cluster. The file also includes a preliminary hidden state representation analysis.


XGBoost:
XGBoost was applied both to the engineered/augmented dataset and the raw dataset. The native XGBoost api was used due to its support for the Quantile data matrix format and quantile regression. 
Note: The current stable release does not support sample weights for quantile regression, however this feature is available in newer versions of the xgboost library. As a result, a recent nightly build was used to fit quantile regression xgb models with weighted samples.
Ultimately, the XGBoost trained on the raw dataset outperformed the competitor! Applying Recursive Feature Elimination (RFE) to the engineered/augmented dataset did not yield significant improvements. All models developed appeared to overfit, so the addition of more features that are also correlated with each other seems to have harmed rather than helped.


MLP:
This was a simple wide MLP with additional learned embeddings for categorical variables that was trained on the engineered/augmented dataset. The model included one output per quantile and was trained with the pinball loss function. Dropout was used for regularization. Various hyper parameter settings (including an optional 2nd layer and layer norm) were tested but did not yield improvements relative to validation set performance compared to a single wide hidden layer with dropout.

Ultimately though the MLP model did not overfit nearly as badly as the XGBoost models, the latter still outperformed the MLP model on the validation and held out test sets.

The predictions from the best fit XGBoost model is included in the root directory as results_xgb_winner.csv in the desired format.
In addition the root directory also include the predictions from the best fit MLP model and is aptly named as results_mlp_loser.csv 
