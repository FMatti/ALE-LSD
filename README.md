# ALE-LSD
Feature importance analysis by accumulated local effects (ALE) in photoacoustic oximetry by learned spectral decoloring (LSD).
## Summary
Based on `numpy` and `matplotllib.pyplot` we implemented a method of determining the impact a feature (i.e. an illumination wavelength) has on the decision process of machine learning regressors in learned spectral decoloring (LDS) and multiple illumination learned spectral decoloring (MI-LSD). The implementation is particularly suited for histogram-based gradient boosters ([LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html), [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html), [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)), because they are usually not too computationally expensive. Using our method of determining a feature's importance, we were able to demonstrate that the absolute prediction errors remain almost unchanged when more than half of the initially available features/wavelengths are removed (i.e. only the 8 or even 6 most important instead of all 16 illumination wavelengths are used for training). Our method also compares to the current approach of uniformally removing features/wavelengths while maintaining the wavelength span width as large as possible [3].
## File structure
* `ale.py` contains the `AccumulatedLocalEffects()` class.
* `examples.ipynb` contains some basic examples based on an example data set stored in the folder `data/`.
* `plots.ipynb` can be used to reproduce all plots displayed in the thesis.
## Data
All rCu data is available at

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4549631.svg)](https://doi.org/10.5281/zenodo.4549631)

## Examples
The following examples serf to briefly demonstrate the most important methods included in the `AccumulatedLocalEffects()` class:
### Feature importance indices
Obtain an ordered list of your features, sorted according to ascending importance in the decision making progress of your regressor.

    feature_importance = array([5,  1, 14,  4, 13,  9, 12,  6,  8, 11,  7,  3, 10,  2, 15,  0])

### Plot the ALE function
This plot shows the accumulated local effects (ALE) function [2] for each wavelength and one illumination position.

![ALE_function example](/plots/EXAMPLE_LSD_ALE_function.PNG)

The same plot, but for multiple illumination positions.

![ALE_function example MI-LSD](/plots/EXAMPLE_MI-LSD_ALE_function.PNG)

### Plot the progression of the absolute errors by 'clipping features'
This plot displays the progression of the absolute error while sequentially excluding a feature from the training process according to some criterion, e.g. with the clipping orders `'state_of_the_art'` [3] (features clipped uniformally), `'min_ALE'` (features clipped according to the feature importances, determined by the total variation [4] of the ALE functions), and `'updated_min_ALE'` (least important feature in every round is removed).

![feature clipping example](/plots/EXAMPLE_LSD_FEATCLIP_state_of_the_art-min_ALE-updated_min_ALE.PNG)

... and now for MI-LSD, with different clipping orders: `'alternating'` (first remove even, then odd feature indices), `'random'` (random clipping order), and `'custom'` (order is passed via the ordered_indices parameter).

![feature clipping example MI-LSD](/plots/EXAMPLE_MI-LSD_FEATCLIP_alternating-random-custom-min_ALE.PNG)

## References
[1] T. Kirchner and M. Frenz (2021). "Quantitative photoacoustic oximetry imaging by multiple illumination learned spectral decoloring". [arXiv:2102.11201v1](https://arxiv.org/abs/2102.11201)

[2] D. W. Apley and J. Zhu (2019). "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models". [arXiv:1612.08468v2](https://arxiv.org/abs/1612.08468)

[3] G. P. Luke, S. Y. Nam, and S. Y. Emelianov (2013). “Optical wavelength selection for
improved spectroscopic photoacoustic imaging”. [https://doi.org/10.1016/j.pacs.2013.08.001](10.1016/j.pacs.2013.08.001)

[4] B. I. Golubov and A. G. Vitushkin (2001). ["Variation of a function". Encyclopedia of Mathematics, EMS Press.](https://encyclopediaofmath.org/index.php?title=Variation_of_a_function)
