___

<!-- Author: https://www.kaggle.com/lasm1984 -->

This notebook analyzes store sales time series data to make predictions for a Kaggle competition [Alexis Cook, DanB, inversion, Ryan Holbrook. (2021). Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). It loads and prepares the data, conducts exploratory analysis, trains Prophet forecasting models on sliced datasets and makes predictions.

Author: [Ale uy](https://www.kaggle.com/lasm1984) 

___

<h1 style="background-color:red;font-family:newtimeroman;color:black;font-size:380%;text-align:center;border-radius: 50px 50px;">Store Sales - Time Series Forecasting</h1>

<a id='goto0'></a>
<h1 style="background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;">Table of Contents</h1>

0. [Table of Contents](#goto0)

1. [Notebook Description](#goto1)

2. [Notebook file](https://github.com/ale-uy/StoreSalesTimeSeriesForecasting/blob/main/StoreSales.ipynb)

3. [Final Project PDF](https://github.com/ale-uy/StoreSalesTimeSeriesForecasting/blob/main/README.pdf)

4. [Conclusions](#goto4)

<a id='goto1'></a>
# <h1 style="background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;">Notebook Description</h1>

[Back](#goto0)

## Description

### Goal of the Competition

I'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores.

### Context

Forecasts aren’t just for meteorologists. Governments forecast economic growth. Scientists attempt to predict the future population. And businesses forecast product demand—a common task of professional data scientists. Forecasts are especially relevant to brick-and-mortar grocery stores, which must dance delicately with how much inventory to buy. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leading to lost revenue and upset customers. More accurate forecasting, thanks to machine learning, could help ensure retailers please customers by having just enough of the right products at the right time.

Current subjective forecasting methods for retail have little data to back them up and are unlikely to be automated. The problem becomes even more complex as retailers add new locations with unique needs, new products, ever-transitioning seasonal tastes, and unpredictable product marketing.

### Potential Impact

If successful, you'll have flexed some new skills in a real world example. For grocery stores, more accurate forecasting can decrease food waste related to overstocking and improve customer satisfaction. The results of this ongoing competition, over time, might even ensure your local store has exactly what you need the next time you shop.

**Author: [Ale uy](https://www.kaggle.com/lasm1984)**

<a id='goto4'></a>
# <h1 style="background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;">Conclusions</h1>

[Back](#goto0)

### Evaluation
The evaluation metric for this competition is **Root Mean Squared Logarithmic Error**.

The RMSLE is calculated as:
$$
\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\log(1+y^i) - \log(1+y_i)\right)^2}
$$

Where: 

$$ 
n - \text{ represents the total number of instances.,}\\
y^i - \text{ is the predicted target value for instance } (i),\\
y_i - \text{ is the actual target value for instance} (i),\\
\log - \text{ denotes the natural logarithm.}
$$

### Submissions Scores

**output.csv:** (with hyperparameter `changepoint_prior_scale = 3`)
* Score (rmsle): 0.46566

This score gives this model a place among the 20% of best performance within the competition.
