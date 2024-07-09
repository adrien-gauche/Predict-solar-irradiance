import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

### TIMES SERIES ###
# Example: target = weight_1 * feature_1 + weight_2 * feature_2 + bias
# regression (ordinary least squares) learns the weights and bias that minimize the mean squared error between the predicted and actual target values.
# weights : regression coefficients
# bias : intercept

# Time-step features: time dependent features, like the day of the week
# with linear regression : target = weight * time + bias
# Lag features: values from previous time steps
# with linear regression: target = weight * lag + bias

# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html


import warnings

import matplotlib.pyplot as plt
from IPython import get_ipython

warnings.simplefilter("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

get_ipython().config.InlineBackend.figure_format = 'retina'


def predict_time_serie(df):
    """
    This function predicts a time series using linear regression.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the time series data.

    Returns:
    pandas.DataFrame: The DataFrame with the predicted values.
    """

    # Create a time column
    df["time"] = np.arange(len(df.index))

    # Create a lag column
    df["lag"] = df["target"].shift(1)

    # Drop missing values
    df.dropna(inplace=True)

    # TREND
    moving_average = df.rolling(
        window=365,  # 365-day window
        center=True,  # puts the average at the center of the window
        min_periods=183,  # choose about half the window size
    ).mean()  # compute the mean (could also do median, std, min, max, ...)

    ax = df.plot(style=".", color="0.5")
    moving_average.plot(
        ax=ax,
        linewidth=3,
        title="Tunnel Traffic - 365-Day Moving Average",
        legend=False,
    )

    # SEASONALITY
    # Two features possibles:
    # Seasonal indicators (one-hot encoding)
    # Fourier features (sine and cosine functions)
    # Compute Fourier features with statsmodels

    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

    # for trend
    dp = DeterministicProcess(
        index=df.index,  # dates from the training data
        constant=True,  # dummy feature for the bias (y_intercept)
        order=1,  # trend (order 1 means linear)
        drop=True,  # drop terms if necessary to avoid collinearity
    )

    # fourier features, for seasonality
    fourier = CalendarFourier(freq="A", order=10)  # ordre = sin/cos pairs
    dp = DeterministicProcess(
        index=df.index,  # dates from the training data
        constant=True,  # dummy feature for the bias (y_intercept)
        seasonal=True,  # weekly seasonality (indicators)
        additional_terms=[fourier],  # annual seasonality (fourier)
        drop=True,  # drop terms to avoid collinearity
    )

    # `in_sample` creates features for the dates given in the `index` argument
    X = dp.in_sample()

    from sklearn.linear_model import LinearRegression

    # Training data
    X = df  # features
    y = X.pop("target")  # target
    
    X = make_lags(y, lags=4).dropna()

    y = make_multistep_target(y, steps=16).dropna()
    


    from sklearn.multioutput import MultiOutputRegressor
    #model = MultiOutputRegressor(XGBRegressor()) # direct strategy, 1 model per time step
    from sklearn.multioutput import RegressorChain #Direct Recursive strategy
    #DirRec strategy capture serial dependence better than Direct, but it can also suffer from error propagation like Recursive.
    #model = RegressorChain(base_estimator=XGBRegressor())
    #from sklearn.linear_model import LinearRegression
    #model = LinearRegression()

    # Useful after dropping missing values
    y, X = y.align(X, join="inner", axis=0)  # drop corresponding values in target

    model.fit(X, y)

    y_pred = pd.DataFrame(
        model.predict(X),
        index=y.index,
        columns=y.columns,
    ).clip(0.0)
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax = y.plot(**plot_params, ax=ax, alpha=0.5)
    ax = plot_multistep(y_pred, ax=ax, every=EVERY)
    _ = ax.legend(['y', 'y' + ' Forecast'])

    return y_pred


def encode_holidays(X, holidays_events):
    # National and regional holidays in the training set
    holidays = (
        holidays_events.query("locale in ['National', 'Regional']")
        .loc["2017":"2017-08-15", ["description"]]
        .assign(description=lambda x: x.description.cat.remove_unused_categories())
    )

    display(holidays)

    # Scikit-learn solution
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(sparse=False)

    X_holidays = pd.DataFrame(
        ohe.fit_transform(holidays),
        index=holidays.index,
        columns=holidays.description.unique(),
    )

    # Pandas solution
    X_holidays = pd.get_dummies(holidays)

    X2 = X.join(X_holidays, on="date").fillna(0.0)


# From Lesson 3
def seasonal_plot(X, y, period, freq, ax=None):
    # If ax is not provided, create a new subplot
    if ax is None:
        _, ax = plt.subplots()

    # Define a color palette for the lines in the plot
    # The number of colors is equal to the number of unique values in X[period]
    palette = sns.color_palette(
        "husl",
        n_colors=X[period].nunique(),
    )

    # Plot the data: x=freq, y=y, hue=period
    # The hue argument is used to specify the categorical variable for differentiating the lines
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,  # do not show confidence intervals
        ax=ax,
        palette=palette,
        legend=False,  # do not show the legend
    )

    # Set the title of the plot
    ax.set_title(f"Seasonal Plot ({period}/{freq})")

    # Annotate the last point of each line with the corresponding value of X[period]
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]  # get the y-coordinate of the last point of the line
        ax.annotate(
            name,  # the text to display
            xy=(
                1,
                y_,
            ),  # the coordinates of the text (1, y_) is the last point of the line
            xytext=(6, 0),  # the offset of the text from the specified coordinates
            color=line.get_color(),  # the color of the text is the same as the color of the line
            xycoords=ax.get_yaxis_transform(),  # the coordinate system to use for xy
            textcoords="offset points",  # the coordinate system to use for xytext
            size=14,  # the font size of the text
            va="center",  # the vertical alignment of the text
        )

    # Return the axes object
    return ax


def plot_periodogram(
    ts, fs=pd.Timedelta("365D") / pd.Timedelta("1D"), detrend="linear", ax=None
):
    # Import the periodogram function from the scipy.signal module
    from scipy.signal import periodogram

    # Calculate the sampling frequency: the number of samples per unit of time
    # In this case, the unit of time is 1 day, and the number of samples is 365 days (1 year)

    # The scaling argument is used to specify the type of scaling to apply to the periodogram
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,  # sampling frequency, the number of samples per unit of time
        detrend=detrend,
        window="boxcar",  # the type of window function to apply to the time series
        scaling="spectrum",  # the type of scaling to apply to the periodogram
    )

    # If ax is not provided, create a new subplot
    if ax is None:
        _, ax = plt.subplots()

    # Plot the periodogram: x=freqencies, y=spectrum
    # The step function is used to create a step plot, which is a type of plot that shows the change in a value over a certain period of time
    ax.step(freqencies, spectrum, color="purple")

    # Set the x-axis to a logarithmic scale
    ax.set_xscale("log")

    # Set the x-axis ticks and labels
    # The ticks are the values at which the labels are placed
    # The labels are the text that is displayed at the ticks
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,  # the angle of rotation of the labels
    )

    # Set the y-axis labels to display in scientific notation
    ax.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0)
    )  # the range of values to display in scientific notation

    # Set the y-axis label
    ax.set_ylabel("Variance")

    # Set the title of the plot
    ax.set_title("Periodogram")

    # Return the axes object
    return ax


# From Lesson 4

from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(y, lags=4*24) # Mooving Average MA (q)

from statsmodels.graphics.tsaplots import plot_pacf
#plot_pacf(y, lags=4*24) # Auto-Regressive AR (p)

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    # Import the AnchoredText class from the matplotlib.offsetbox module
    from matplotlib.offsetbox import AnchoredText

    # Shift the time series x by the specified lag
    x_ = x.shift(lag)

    # If standardize is True, scale the time series x_ to have zero mean and unit variance
    if standardize:
        # Import the StandardScaler class from the sklearn.preprocessing module
        from sklearn.preprocessing import StandardScaler

        # Initialize a new StandardScaler object
        scaler = StandardScaler()
        # Scale the time series x_ using the fit_transform method of the StandardScaler object
        x_ = pd.DataFrame(scaler.fit_transform(x_), index=x_.index, columns=x_.columns)

    # If y is not None, scale it using the same StandardScaler object (if standardize is True)
    if y is not None:
        if standardize:
            y_ = pd.DataFrame(scaler.transform(y), index=y.index, columns=y.columns)
        else:
            y_ = y
    else:
        y_ = x

    # Calculate the correlation between the time series x_ and y_
    corr = y_.corr(x_)

    # If ax is not provided, create a new subplot
    if ax is None:
        fig, ax = plt.subplots()

    # Define the keyword arguments for the scatter plot
    scatter_kws = dict(
        alpha=0.75,  # the transparency of the markers
        s=3,  # the size of the markers
    )

    # Define the keyword arguments for the regression line
    line_kws = dict(
        color="C3",
    )  # the color of the line

    # Plot the data: x=x_, y=y_
    # The regplot function is used to create a scatter plot with a regression line
    ax = sns.regplot(
        x=x_,
        y=y_,
        scatter_kws=scatter_kws,  # the keyword arguments for the scatter plot
        line_kws=line_kws,  # the keyword arguments for the regression line
        lowess=True,  # use a LOWESS regression instead of a linear regression
        ax=ax,
        **kwargs,
    )  # any additional keyword arguments

    # Add a text box to the plot with the correlation coefficient
    # The AnchoredText class is used to create a text box that is anchored to a specific location in the plot
    at = AnchoredText(
        f"{corr:.2f}",  # the text to display
        prop=dict(size="large"),  # the font size of the text
        frameon=True,  # whether to draw a frame around the text
        loc="upper left",  # the location of the text box
    )
    # Set the style of the frame around the text
    at.patch.set_boxstyle("square, pad=0.0")
    # Add the text box to the plot
    ax.add_artist(at)

    # Set the title, x-axis label, and y-axis label of the plot
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)

    # Return the axes object
    return ax


def plot_lags(
    x,
    y=None,
    lags=6,  # the number of lags to plot
    leads=None,  # the number of leads to plot (if not provided, defaults to 0)
    nrows=1,  # the number of rows in the plot grid
    lagplot_kwargs={},  # any additional keyword arguments to pass to the lagplot function
    **kwargs,
):  # any additional keyword arguments

    # Import the math module
    import math

    # If nrows is not provided in kwargs, set the default value to nrows
    kwargs.setdefault("nrows", nrows)

    # If leads is not provided, set the default value to 0
    leads = leads or 0

    # If ncols is not provided in kwargs, set the default value to the smallest integer greater than or equal to (lags + orig + leads) / nrows
    kwargs.setdefault("ncols", math.ceil((lags + orig + leads) / nrows))

    # If figsize is not provided in kwargs, set the default value to (kwargs['ncols'] * 2, nrows * 2 + 0.5)
    kwargs.setdefault("figsize", (kwargs["ncols"] * 2, nrows * 2 + 0.5))

    # Create a new figure and a grid of subplots
    # The sharex and sharey arguments are used to specify whether the x-axis and y-axis, respectively, should be shared among all subplots
    # The squeeze argument is used to specify whether to return a single axes object if only one subplot is created
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)

    # Loop through each subplot and plot the data
    # The get_axes method is used to get a list of all axes objects in the figure
    for ax, k in zip(fig.get_axes(), range(kwargs["nrows"] * kwargs["ncols"])):
        # Subtract the number of leads and orig (if not provided, defaults to 1) from k
        k -= leads + orig

        # If k + 1 is less than or equal to lags, plot the data using the lagplot function
        if k + 1 <= lags:
            ax = lagplot(x, y, shift=k + 1, ax=ax, **lagplot_kwargs)
            # Set the title of the subplot
            title = f"Lag {k + 1}" if k + 1 >= 0 else f"Lead {-k - 1}"
            ax.set_title(title, fontdict=dict(fontsize=14))
            # Set the x-axis and y-axis labels to be empty
            ax.set(xlabel="", ylabel="")
        else:
            # If k + 1 is greater than lags, turn off the axes of the subplot
            ax.axis("off")

    # Set the x-axis labels for the last row of subplots
    plt.setp(axs[-1, :], xlabel=x.name)

    # Set the y-axis labels for the first column of subplots
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)

    # Adjust the spacing between the subplots
    # The w_pad and h_pad arguments are used to specify the padding between the subplots, in inches
    fig.tight_layout(w_pad=0.1, h_pad=0.1)

    # Return the figure object
    return fig


# From Lesson 5
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        # Initialize the first model
        self.model_1 = model_1
        # Initialize the second model
        self.model_2 = model_2
        # Initialize the attribute to store the columns of y (target)
        self.y_columns = None
        # Initialize the attribute to store the stack_cols
        self.stack_cols = None

    def fit(self, X_1, X_2, y, stack_cols=None):
        # Train the first model on X_1 and y
        self.model_1.fit(X_1, y)

        # Make predictions with the first model on X_1
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=y.columns,
        )
        # Compute the residuals: y - y_fit
        y_resid = y - y_fit
        # Reshape the residuals from wide to long format
        y_resid = y_resid.stack(stack_cols).squeeze()

        # Train the second model on X_2 and the residuals
        self.model_2.fit(X_2, y_resid)

        # Store the columns of y and stack_cols for the predict method
        self.y_columns = y.columns
        self.stack_cols = stack_cols

    def predict(self, X_1, X_2):
        # Make predictions with the first model on X_1
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=self.y_columns,
        )
        # Reshape the predictions from wide to long format
        y_pred = y_pred.stack(self.stack_cols).squeeze()

        # Add the predictions of the second model to the predictions of the first model
        y_pred += self.model_2.predict(X_2)
        # Return the predictions in the original wide format
        return y_pred.unstack(self.stack_cols)


# From Lesson 6
# Function to create lagged features for a time series
def make_lags(ts, lags, lead_time=1, name="y"):
    """ Use dropna() or fillna() to handle missing values"""
    # Concatenate the time series shifted by different lags, along the columns axis
    # The keys of the dictionary are the names of the columns in the resulting DataFrame
    return pd.concat(
        {
            f"{name}_lag_{i}": ts.shift(i)  # shift the time series by i periods
            for i in range(
                lead_time, lags + lead_time
            )  # generate the lags from lead_time to lags + lead_time
        },
        axis=1,  # concatenate along the columns axis
    )


# Function to create lead features for a time series
def make_leads(ts, leads, name="y"):
    # Concatenate the time series shifted by different leads, along the columns axis
    return pd.concat(
        {
            f"{name}_lead_{i}": ts.shift(
                -i
            )  # shift the time series by -i periods (i.e., into the future)
            for i in reversed(range(leads))
        },  # generate the leads from leads - 1 to 0
        axis=1,  # concatenate along the columns axis
    )


def make_multistep_target(ts, steps, reverse=False):
    """Generate multi-step target for a time series.
    Args:
        ts (DataFrame): Time series data.
        steps (int): Number of steps to forecast.
        reverse (bool, optional): If True, generate the shifts from steps - 1 to 0. Defaults to False.

    Returns:
        DataFrame: 
    """
    # Generate the shifts for the multi-step target
    # If reverse is True, the shifts are generated from steps - 1 to 0, otherwise from 0 to steps - 1
    shifts = reversed(range(steps)) if reverse else range(steps)
    # Concatenate the time series shifted by different steps, along the columns axis
    # The keys of the dictionary are the names of the columns in the resulting DataFrame
    return pd.concat(
        {
            f"y_step_{i + 1}": ts.shift(
                -i
            )  # shift the time series by -i periods (i.e., into the future)
            for i in shifts
        },  # generate the shifts
        axis=1,
    )  # concatenate along the columns axis

# Function to plot the multi-step predictions for a time series
def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    # Define the default values for the color palette
    palette_kwargs_ = dict(palette="husl", n_colors=16, desat=None)
    # If palette_kwargs is not None, update the default values with the values in palette_kwargs
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    # Generate the color palette
    palette = sns.color_palette(**palette_kwargs_)
    # If ax is None, create a new subplot
    if ax is None:
        fig, ax = plt.subplots()
    # Set the color cycle for the subplot
    ax.set_prop_cycle(plt.cycler("color", palette))
    
    freq = y.index.freq
    if freq is None:
        freq = pd.infer_freq(y.index)
        
    # Loop through the rows of the DataFrame, with a step size of every
    for datetime, preds in y[::every].iterrows():
        # Set the index of the predictions to be a range of periods, starting from the date
        preds.index = pd.period_range(start=datetime, periods=len(preds), freq=freq)
        # Plot the predictions on the subplot
        preds.plot(ax=ax)
    # Return the subplot
    return ax