# ALE-LSD
Feature importance analysis by accumulated local effects (ALE) in photoacoustic oximetry by learned spectral decoloring (LSD).
## Summary
Based on `numpy` and `matplotllib.pyplot` we implemented a method of determining the impact a feature (i.e. an illumination wavelength) has on the decision process of machine learning regressors. The implementation is particularly suited for histogram based gradient boosters ([LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html), [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html), [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)), because they are usually not too computationally expensive. Using our method of determining a feature's importance, we were able to demonstrate that the absolute prediction errors, and the median absolute prediction error in particular, remain almost unchanged when more than half of the initially available features/wavelengths are removed (i.e. only the 6 most important instead of all 16 illumination wavelengths are used for training). Our method also outperforms the current approach of uniformally removing features/wavelengths while maintaining the wavelength span width as large as possible.
## File structure
* `ale.py` contains the `AccumulatedLocalEffects()` class.
* `examples.ipynb` contains some basic examples based on an example data set stored in the folder `data`.
* `plots.ipynb` can be used to reproduce all plots displayed in the paper and stored in the folder `plots`.
## Examples
The following examples serf to quickly demonstrate the most important methods included in the `AccumulatedLocalEffects()` class:
### Feature importance indices
Obtain an ordered list of your features, sorted according to ascending importance in the decision making progress of your regressor.

    feature_importance = [ 1  5  4 14 13 12  9  8  6 11  7  3 10  2 15  0]

### Plot the ALE function
This plot shows the accumulated local effects (ALE) function [2] for each wavelength and illumination position. It visually confirms the validity of our approach to determining feature importances.
![ALE_function example](/plots/EXAMPLE_ALE_function.png)
### Plot the progression of the absolute errors while doing 'clipping features'
This plot displays the progression of the absolute error while sequentially excluding a feature from the training process according to some criterion, e.g. the `importance` determined using the approximate total variation [3] of each feature, or `uniformal` which removes features uniformally. It demonstrates the effectiveness of our method of evaluating the importance of a feature.
![feature clipping example](/plots/EXAMPLE_FEATCLIP_importance-uniformal.png)
## Resources
[1] Kirchner, Thomas & Frenz, Martin (2021). "Quantitative photoacoustic oximetry imaging by multiple illumination learned spectral decoloring". [arXiv:2102.11201v1](https://arxiv.org/abs/2102.11201)

[2] Apley, Daniel W. & Zhu, Jingyu (2019). "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models". [arXiv:1612.08468v2](https://arxiv.org/abs/1612.08468)

[3] Golubov, Boris I. & Vitushkin, Anatoli G. (2001). ["Variation of a function". Encyclopedia of Mathematics, EMS Press.](https://encyclopediaofmath.org/index.php?title=Variation_of_a_function)
