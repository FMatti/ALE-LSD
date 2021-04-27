import numpy as np
from matplotlib import pyplot as plt

# LaTeX typesetting
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{eulervm}",
    "font.family": "serif",
    "font.serif": "Palatino",
})


class AccumulatedLocalEffects:
    """
    This class allows you to analyze and use the local accumulated effects [2]
    function value in order to optimize feature selection in learned spectral
    decoloring (LSD) [1].

    You'll need to provide this class with a machine learning regressor 'reg'
    and a training data set (X_train, y_train) to fit the regressor with.

    References
    ----------
    [1] Kirchner, Thomas & Frenz, Martin (2021). "Quantitative photoacoustic
        oximetry imaging by multiple illumination learned spectral decoloring".
        arXiv:2102.11201v1
    [2] Apley, Daniel W. & Zhu, Jingyu (2019). "Visualizing the Effects of
        Predictor Variables in Black Box Supervised Learning Models".
        arXiv:1612.08468v2.
    [3] Golubov, Boris I. & Vitushkin, Anatoli G. (2001).
        "Variation of a function". Encyclopedia of Mathematics, EMS Press.

    Parameters
    ----------
    reg : object
        REGRESSOR TO BE USED.
    filename : str
        FILENAME UNDER WHICH THE PLOTS AND REGRESSORS WILL BE SAVED.
    X_train : Array of float (N, M)
        SAMPLE-FEATURE MATRIX FOR TRAINING THE REGRESSORS.
    Y_train : Array of float (N)
        LABELS FOR TRAINING THE REGRESSORS.
    num_illum : int32
        NUMBER OF ILLUMINATION POSITIONS (USE 1 FOR LSD).
    num_wlen : int32
        NUMBER OF WAVELENGTHS USED FOR ILLUMINATION.
    data_reduction : float64, optional (default is 0.01)
        SUBRATIO OF THE DATA TO BE USED FOR CALCULATING THE f_ALE FUNCTION.
    num_subintervals : int32, optional (default is 100)
        NUMBER OF UNIFORM INTERVALS FOR THE FEATURES TO BE SUBDIVIDED INTO.
    num_x_values : int32, optional (default is 20)
        NUMBER OF x-VALUES FOR THE f_ALE TO BE EVALUATED AT.

    Methods
    -------
    feature_importance_indices(illum_pos, features)
        RETURNS SORTED INDICES OF FEATURES BY IMPORTANCE (ASCENDING).
    plot_ALE_function(illum_pos)
        PLOTS THE ALE FUNCTIONS FOR A SPECIFIED ILLUMINATION POSITION.
    plot_feature_clipping(
        X_test, y_test, clipping_order, illum_pos, ordered_indices, seed)
        PLOTS THE PROGRESSION OF THE MEDIAN ABSOLUTE ERROR IN FEATURE CLIPPING.

    """

    def __init__(self, reg, filename, X_train, y_train, num_illum, num_wlen,
                 num_subintervals=100, num_x_values=20, data_reduction=0.01):

        self.reg = reg
        self.filename = filename
        self.y_train = y_train
        self.X_train = self._normalize_features(X_train, num_illum)
        self.num_illum = num_illum
        self.num_wlen = num_wlen
        self.num_subintervals = num_subintervals
        self.num_x_values = num_x_values
        self.data_reduction = data_reduction

    @staticmethod
    def _normalize_features(X, n_illum):
        """
        Takes the L1-Norm of every set of n_wlen wavelengths at n_illum
        different illumination positions, and uses it in order to normalize
        each row to a row-sum of 1.

        Parameters:
        -----------
        X : Array of float64 (N,n_illum*n_wlen)
            RAW DATA SET.
        n_illum : int32
            NUMBER OF ILLUMINATION POSITIONS IN DATA SET.

        Returns:
        --------
        X : Array of float64 (N,n_illum*n_wlen)
            L1-NORMALIZED DATA SET.

        """

        # Obtaining the number of features per illumination position
        n_wlen = int(len(X[0, :]) / n_illum)

        # Iterating through all illumination positions
        for i in range(n_illum):

            # Taking the L1 row-norm
            row_sum = np.sum(X[:, i*n_wlen:(i+1)*n_wlen], axis=1)

            # Dividing each row by its corresponding L1 row-norm
            X[:, i*n_wlen:(i+1)*n_wlen] /= row_sum.reshape((len(X[:, 0]), 1))

        return X

    def _mean_pred_diff(self, feat, X, a, b):
        """
        Calculates mean difference in prediction for the samples of a
        feature, which have values within the interval [a,b]. First, all such
        sample-values for this feature are replaced by a, and predictions are
        made. Subsequently, they are replaced by b, and once again predictions
        are made. The mean difference in these two predictions is returned.

        Parameters
        ----------
        feat : int32
            FEATURE FOR WHICH WILL BE LOOKED AT.
        X : Array of float64 (N, M)
            SAMPLE-FEATURE MATRIX.
        a : float64
            LOWER INTERVAL BOUNDARY.
        b : float64
            UPPER INTERVAL BOUNDARY.

        Returns
        -------
         mean_pred_diff : float64
            MEAN PREDICTION DIFFERENCE.

        """

        # Only taking the samples with values of this feature in the interval
        X_in_interval = X[(a < X[:, feat]) & (X[:, feat] <= b)]

        # Predictions for values of this feature replaced by interval border a
        X_in_interval[:, feat] = a
        y_pred_lower = self.reg.predict(X_in_interval)

        # Predictions for values of this feature replaced by interval border b
        X_in_interval[:, feat] = b
        y_pred_upper = self.reg.predict(X_in_interval)

        # Calculating the mean over all the prediction differences
        mean_pred_diff = np.mean(y_pred_upper - y_pred_lower)

        return mean_pred_diff

    def _ALE_function(self, feat, X):
        """
        Calculates the (centered) ALE function value for a given feature at
        num_x_values evenly spaced points, ranging from the smallest to the
        largest attained value in this feature. The ALE at a point x is the
        sum of all the mean prediction differences that are made when replacing
        each sample for this feature within a given subinterval first by the
        upper subinterval bound, and then by the lower bound. The subintervals
        are chosen such that each contains an equal amount of samples. [2]

        Parameters
        ----------
        feat : int32
            FEATURE, FOR WHICH THE f_ALE FUNCTION WILL BE CALCULATED.
        X : Array of float64 (N, M)
            SAMPLE-FEATURE MATRIX.

        Returns
        -------
        ALE_function : Array of float64 (num_x_values)
            f_ALE EVALUATED AT num_x_values EVENLY SPACED POINTS.
        x_values : Array of float64 (num_x_values)
            THE num_x_values EVENLY SPACED POINTS, WHERE f_ALE IS EVALUATED AT.

        """

        # Uniformly reducing the amount of data to be used in calculation
        _X = X[::int(1/self.data_reduction), :]

        # Partitioning the values for this feature into subintervals, spaced
        # according to uniform quantiles of the empirical distribution function
        partition = np.quantile(
            _X[:, feat], np.linspace(0, 1, self.num_subintervals + 1))

        # Lowering smallest partition, such that it includes the smallest value
        partition[0] -= 1e-5

        # Uniformly spaced points spanning all attained values, for which
        # we'll calculate and return f_ALE(x_values)
        x_values = np.linspace(
            np.min(_X[:, feat]), np.max(_X[:, feat]), self.num_x_values)

        # Initializing some variables to keep track of the f_ALE
        ALE_function = np.empty(self.num_x_values)
        f_ALE = 0
        f_ALE_sum = 0

        # Index to keep track of x_values that already have an f_ALE assigned
        j = 0

        # Iterating through all subintervals
        for i in range(self.num_subintervals):

            # Accumulating the prediction differences for this subintervals
            f_ALE += self._mean_pred_diff(
                feat, _X, partition[i], partition[i+1])

            # Summing the f_ALE for using them in the centralization later on
            f_ALE_sum += f_ALE

            # Checking whether any of our x_values fall into this subinterval
            while x_values[j] <= partition[i+1]:

                # If yes, the f_ALE for this x_value is recorded
                ALE_function[j] = f_ALE

                # Increasing the index, when an x_value was in the subinterval
                j += 1

                # Stopping the iteration when arrived at last entry in x_values
                if j == self.num_x_values:

                    break

        # Centralizing the f_ALE function with the average of all attained
        # f_ALE values at subintervals borders
        ALE_function -= f_ALE_sum / (self.num_subintervals + 1)

        return ALE_function, x_values

    def _ALE_total_variation(self, feat, X):
        """
        Calculating the approximate total variation [3] of the f_ALE function
        for one features, by summing the absolute differences between the f_ALE
        function values at uniformly spaced points. This loosely corresponds
        to the total effect this feature has on the predictions.

        Parameters
        ----------
        feat : int32
            FEATURE FOR WHICH THE TOTAL VARIATION IS CALCULATED.
        X : Array of float64 (N, M)
            SAMPLE-FEATURE MATRIX.

        Returns
        -------
         f_ALE_var : float64
            TOTAL VARIATION OF ALL CALCULATED f_ALE VALUES FOR A FEATURE.

        """

        # Calling the ALE function
        ALE_function = self._ALE_function(feat, X)[0]

        # Calculating an approximation of the total variation of this function
        ALE_var = sum([abs(ALE_function[i+1] - ALE_function[i])
                       for i in range(self.num_x_values - 1)])

        return ALE_var

    def feature_importance_indices(self, illum_pos=[0], features=None):
        """
        Ranks the features of the specified illumination position in X_train
        according to their importance (i.e. the higher a feature's ALE total
        variation is, the more impact it'll have on the predictions), and
        returns a sorted list (least important first) of the feature indices.

        Parameters
        ----------
        illum_pos : List of int32, optional (default is [0])
            ILLUMINATION POSITIONS TO USE IN CALCULATION OF THE IMPORTANCE.
        features : List of int32, optional (default is None, i.e. all features)
            FEATURES (WAVELENGTHS) TO USE IN CALCULATION OF THE IMPORTANCE.

        Returns
        -------
        ALE_var_ordered_indices : Array of float64 (M)
            INDICES OF FEATURES SORTED BY THEIR ALE TOTAL VARIATION.

        """

        # If no features were specified, using all features/wavelengths
        if features is None:

            features = np.arange(self.num_wlen)

        # Declaring a variable to keep track of the total variation of the ALE
        ALE_var = np.zeros(len(features))

        # Adding up the ALE variations for the given illumination positions
        for i in illum_pos:

            # Fitting the regressor
            self.reg.fit(
                self._normalize_features(
                    self.X_train[:, i*len(features) + features], n_illum=1),
                self.y_train)

            # Indices of features sorted by their f_ALE standard deviations
            ALE_var += [self._ALE_total_variation(
                feat, self._normalize_features(
                    self.X_train[:, i*len(features) + features], n_illum=1))
                for feat in range(len(features))]

        # Sorting the features according to their ALE variation (ascending)
        ALE_var_ordered_indices = features[np.argsort(ALE_var)]

        return ALE_var_ordered_indices

    def plot_ALE_function(self, illum_pos=[0]):
        """
        Plot the ALE function for all wavelengths and the specified
        illumination positions.

        Parameters
        ----------
        illum_pos : List of int32, optional (default is [0])
            ILLUMINATION POSITIONS TO BE INCLUDE IN THE ALE FUNCTION PLOT.

        """

        # Initializing the plot-surface
        fig, ax = plt.subplots(
            -(- self.num_wlen // 4), 4, figsize=(7, 7), sharey=True)

        # Defining a bunch of colors
        colors = [[0.106, 0.169, 0.514, 1],
                  [0.209, 0.509, 0.722, 1],
                  [0.506, 0.796, 0.816, 1],
                  [0.835, 0.918, 0.784, 1]]

        for i in illum_pos:

            # Only using features corresponding to the illumination position
            _X_train = self.X_train[
                :, self.num_wlen*i:self.num_wlen*(i + 1)]

            # Fitting the regressor
            self.reg.fit(_X_train, self.y_train)

            # Iterating over all features to generate the plot-grid
            for feat in range(self.num_wlen):

                # Obtaining the approximated ALE_function at x_values
                ALE_function, x_values = self._ALE_function(feat, _X_train)

                # Plotting the y=0 axis for better visibility
                ax[int(feat/4), feat % 4].axhline(
                    y=0, c='k', alpha=0.2, linewidth=1)

                # Plotting the resulting ALE_function values at x_values
                ax[int(feat/4), feat % 4].plot(
                    x_values, ALE_function, c=colors[illum_pos.index(i)])

        # Iterating over all features to specify the dimensions and add labels
        for feat in range(self.num_wlen):

            # Adding a textbox to display the wavelength
            ax[int(feat/4), feat % 4].text(
                0.95, 0.92, r"$\lambda$ = %d nm" % (680+feat*20),
                fontsize='small', ha='right', va='top',
                transform=ax[int(feat/4), feat % 4].transAxes)

            # Adding quantiles if only one illumination position is specified
            if len(illum_pos) == 1:

                # Extracting the quantiles from the empirical distribution
                quantiles = np.quantile(
                    _X_train[:, feat], np.linspace(0, 1, 11)[1:10])

                # Adding quantiles as minor ticks to the x-axis
                ax[int(feat/4), feat % 4].set_xticks(quantiles, minor=True)

                # Adjusting the minor ticks for good looks
                ax[int(feat/4), feat % 4].tick_params(
                    which='minor', direction='in', length=4, colors='k')

        # Creating a legend for the illumination positions if multiple
        # illumination positions were specified
        if len(illum_pos) > 1:

            # Creating dummy-plots for use in the legend
            for i in illum_pos:

                plt.plot([], c=colors[illum_pos.index(i)], label="%d" % i)

            # Adding legend for the illumination positions
            plt.legend(
                title="Illumination position", loc='upper center',
                bbox_to_anchor=[0.5, 1], bbox_transform=plt.gcf().transFigure,
                ncol=len(illum_pos))

        # Adding the global x- and y-label (using textboxes because it works)
        fig.text(0.5, 0.05,
                 r"L1 normalized absorbed energy $H(\mathbf{x_0},\lambda)$",
                 ha='center', va='bottom')
        fig.text(0.05, 0.5, r"Accumulated local effects $f_{ALE}(H)$",
                 ha='left', va='center', rotation=90)

        # Saving the figure
        plt.savefig(self.filename + '_ALE_function.png', dpi=300)

    def plot_feature_clipping(self, X, y, clipping_order, illum_pos=[0],
                              ordered_indices=None, seed=1):
        """
        Creates a boxplot for the absolute prediction errors when performing
        the 'feature clipping' operation according to various clipping orders
        (i.e. order according to which the features are removed sequentially).

        Parameters
        ----------
        X : Array of float64 (N, M)
            SAMPLE-FEATURE MATRIX.
        y : Array of float64 (N)
            LABELS.
        clipping_order : List of str (max. 4 methods)
            RULE, ACCORDING TO WHICH FEATURES ARE SEQUENTIALLY REMOVED.
        illum_pos : List of int32, optional (default is [0])
            ILLUMINATION POSITIONS WHICH WILL BE USED TO DETERMINE IMPORTANCE.
        ordered_indices : List of int32, optional (default is None)
            ORDER OF REMOVING FEATURES IF 'custom' CLIPPING MODE WAS PICKED.
        seed : int32 or None, optional (default is 1)
            SEED FOR GENERATING RANDOM FEATURE INDICES.

        Returns
        -------
        stats : Array of float64 (6, 16)
            HOLDS FOR EACH 'FEATURE CLIPPING' STEP THE FOLLOWING INFORMATION:
                [feature last used in this step, P10, Q1, Q2, Q3, P90]

        """

        # Initializing the arrays which keeps track of some statistical data
        stats = dict()
        y_abserr = dict()

        for c in clipping_order:

            if c == 'random':

                # Setting the seed for randomly picking feature indices
                np.random.seed(seed)

                # Randomly generated feature-indices
                ordered_indices = np.random.choice(
                    np.arange(self.num_wlen),
                    size=self.num_wlen,
                    replace=False)

            elif c == 'uniformal':

                # Uniformally spaced feature-indices
                ordered_indices = np.array([7, 8, 3, 12, 5, 10, 1, 14,
                                           6, 9, 4, 11, 2, 13, 0, 15])

                if self.num_wlen != 16:

                    c = 'importance'

                    print("WARNING! Order 'uniformal' only works for 16")
                    print("         wavelengths. Using 'importance' instead.")

            elif c == 'alternating':

                # Alternating feature-indices (first odd, then even)
                ordered_indices = np.concatenate(
                    (np.arange(self.num_wlen)[1::2],
                     np.arange(self.num_wlen)[::2]))

            elif c == 'custom':

                # Custom indices according to the 'clipping_order' parameter
                ordered_indices = np.array(ordered_indices)

            elif c == 'updatedimportance':

                # The initial order doesn't matter, it will be recalculated
                ordered_indices = np.arange(self.num_wlen)

            else:

                # Obtaining the list of indices of features with lowest impact
                # (i.e. lowest f_ALE standard deviation) first, based on the
                # f_ALE function calculated for the X_train data set.
                ordered_indices = self.feature_importance_indices(illum_pos)

                if c != 'importance':

                    print("WARNING! Invalid clipping order specified. Using")
                    print("         clipping_order='importance' instead.")

            # Initializing variables to keep track of various statistics
            stats[c] = np.empty((6, self.num_wlen))
            y_abserr[c] = []

            # Iteratively removing a feature from sorted_indices, and only
            # using the remaining features for fitting and predicting
            for i in range(self.num_wlen):

                # Taking only the indices into account (all illum_pos)
                # which have biggest impact on predictions (highest f_ALE_std)
                indices = np.empty((self.num_wlen-i)*self.num_illum, dtype=int)

                for j in range(self.num_illum):

                    indices[
                        j*(self.num_wlen-i):(j+1)*(self.num_wlen-i)
                        ] = ordered_indices[i:] + j*self.num_wlen

                # Only using the features with highest f_ALE standard deviation
                _X_train = self._normalize_features(
                    self.X_train[:, indices], n_illum=self.num_illum)
                _X = self._normalize_features(
                    X[:, indices], n_illum=self.num_illum)

                # Fitting the regressor
                self.reg.fit(_X_train, self.y_train)

                # Obtaining the absolute errors and the median absolute error
                y_pred = self.reg.predict(_X)
                y_abserr[c].append(abs(y_pred - y) * 100)
                stats[c][1:, self.num_wlen-1-i] = np.quantile(
                    abs(y_pred - y) * 100, [0.1, 0.25, 0.5, 0.75, 0.9])

                # For 'updatedimportance' order, recalculate importance order
                if (c == 'updatedimportance') & (i < self.num_wlen - 1):

                    ordered_indices[i:] = self.feature_importance_indices(
                        illum_pos, ordered_indices[i:])

            # Adding the ordered indices to the 'stats' array
            stats[c][0, :] = np.flip(ordered_indices)

        # Initializing the plot surface
        plt.figure(figsize=(11, 5))
        plt.yscale('log')

        # The number of clipping orders that were specified
        length = len(clipping_order)

        # Calculating the offsets for displaying multiple clipping modes
        offset = (np.arange(length) - (length - 1)/2) / (length + 0.5)

        # Defining a bunch of colors
        colors = [[0.106, 0.169, 0.514, 0.5],
                  [0.209, 0.509, 0.722, 0.5],
                  [0.506, 0.796, 0.816, 0.5],
                  [0.835, 0.918, 0.784, 0.5]]

        for i, c in zip(range(length), clipping_order):

            # Creating the boxplot
            plt.boxplot(
                y_abserr[c][::-1],
                positions=np.arange(1, self.num_wlen+1) + offset[i],
                whis=[10, 90],
                showfliers=False,
                patch_artist=True,
                widths=1/(length + 2),
                boxprops=dict(facecolor=colors[i]),
                capprops=dict(linewidth=1/(length + 2)),
                medianprops=dict(color='k'))

            # Creating some empty plots for use in the legend later on
            plt.plot([], color=colors[i], label=c, linewidth=5)

        # Adding grid lines to guide the eye
        plt.grid(which='major', axis='y', linewidth=0.5, c='k', alpha=0.5)
        plt.grid(which='minor', axis='y', linewidth=0.25, c='k', alpha=0.5)

        # Adding additional stuff to the plots
        plt.xlabel(r"Number of features used for fit and predictions $~n$")
        plt.ylabel(r"Absolute prediction-errors $~|r_{pred}|$ [pp]")
        plt.xticks(
            np.arange(1, self.num_wlen + 1), np.arange(1, self.num_wlen + 1))
        plt.yticks([1, 10], [1, 10])

        # Adding a legend
        plt.legend(title="Clipping order")

        # Saving the figure
        plt.savefig(self.filename + '_FEATCLIP_' + '-'.join(clipping_order)
                    + '.png', dpi=300)

        return stats
