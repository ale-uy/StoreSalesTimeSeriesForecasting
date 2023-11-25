{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "\n",
    "<!-- Author: https://www.kaggle.com/lasm1984 -->"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This notebook analyzes store sales time series data to make predictions for a Kaggle competition [Alexis Cook, DanB, inversion, Ryan Holbrook. (2021). Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). It loads and prepares the data, conducts exploratory analysis, trains Prophet forecasting models on sliced datasets and makes predictions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Author: [Ale uy](https://www.kaggle.com/lasm1984) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1 style=\"background-color:red;font-family:newtimeroman;color:black;font-size:380%;text-align:center;border-radius: 50px 50px;\">Store Sales - Time Series Forecasting</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto0'></a>\n",
    "<h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Table of Contents</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "0. [Table of Contents](#goto0)\n",
    "\n",
    "1. [Notebook Description](#goto1)\n",
    "\n",
    "2. [Loading Libraries](#goto2)\n",
    "\n",
    "3. [Reading and Join Data Files](#goto3)\n",
    "\n",
    "4. [Data Exploration](#goto4)\n",
    "\n",
    "5. [Data Modeling](#goto5)\n",
    "\n",
    "    5a. [Preprocessing](#goto5a)\n",
    "\n",
    "    5b. [Prophet Model](#goto5b)\n",
    "\n",
    "    5c. [Final Touches](#goto5c)\n",
    "\n",
    "6. [Conclusions](#goto6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto1'></a>\n",
    "# <h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Notebook Description</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back](#goto0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbe5a8f0",
   "metadata": {
    "papermill": {
     "duration": 0.023917,
     "end_time": "2023-10-28T13:31:50.317372",
     "exception": false,
     "start_time": "2023-10-28T13:31:50.293455",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Description\n",
    "\n",
    "### Goal of the Competition\n",
    "\n",
    "In this “getting started” competition, you’ll use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer.\n",
    "\n",
    "Specifically, you'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores. You'll practice your machine learning skills with an approachable training dataset of dates, store, and item information, promotions, and unit sales.\n",
    "\n",
    "### Context\n",
    "\n",
    "Forecasts aren’t just for meteorologists. Governments forecast economic growth. Scientists attempt to predict the future population. And businesses forecast product demand—a common task of professional data scientists. Forecasts are especially relevant to brick-and-mortar grocery stores, which must dance delicately with how much inventory to buy. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leading to lost revenue and upset customers. More accurate forecasting, thanks to machine learning, could help ensure retailers please customers by having just enough of the right products at the right time.\n",
    "\n",
    "Current subjective forecasting methods for retail have little data to back them up and are unlikely to be automated. The problem becomes even more complex as retailers add new locations with unique needs, new products, ever-transitioning seasonal tastes, and unpredictable product marketing.\n",
    "\n",
    "### Potential Impact\n",
    "\n",
    "If successful, you'll have flexed some new skills in a real world example. For grocery stores, more accurate forecasting can decrease food waste related to overstocking and improve customer satisfaction. The results of this ongoing competition, over time, might even ensure your local store has exactly what you need the next time you shop."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Author: [Ale uy](https://www.kaggle.com/lasm1984)**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto2'></a>\n",
    "# <h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Loading Libraries</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back](#goto0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b11bde2a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-10-28T13:31:50.367167Z",
     "iopub.status.busy": "2023-10-28T13:31:50.366385Z",
     "iopub.status.idle": "2023-10-28T13:31:58.275327Z",
     "shell.execute_reply": "2023-10-28T13:31:58.273826Z"
    },
    "papermill": {
     "duration": 7.937663,
     "end_time": "2023-10-28T13:31:58.278604",
     "exception": false,
     "start_time": "2023-10-28T13:31:50.340941",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd; pd.set_option('display.max_columns', 30)\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt; plt.style.use('ggplot')\n",
    "import seaborn as sns\n",
    "import plotly.express as px\n",
    "\n",
    "import warnings; warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b2e17ef",
   "metadata": {
    "papermill": {
     "duration": 0.023527,
     "end_time": "2023-10-28T13:31:58.326260",
     "exception": false,
     "start_time": "2023-10-28T13:31:58.302733",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<a id='goto3'></a>\n",
    "# <h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Reading and Join Data Files</h1> "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back](#goto0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import the main analysis datasets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We check the shape of the training and test dataframes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e366bb38",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-10-28T13:31:58.376409Z",
     "iopub.status.busy": "2023-10-28T13:31:58.375925Z",
     "iopub.status.idle": "2023-10-28T13:31:59.684035Z",
     "shell.execute_reply": "2023-10-28T13:31:59.682183Z"
    },
    "papermill": {
     "duration": 1.337176,
     "end_time": "2023-10-28T13:31:59.687899",
     "exception": false,
     "start_time": "2023-10-28T13:31:58.350723",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train = pd.read_csv('train.csv')\n",
    "test = pd.read_csv('test.csv')\n",
    "\n",
    "print('The dimension of the train dataset is:', train.shape)\n",
    "print('The dimension of the test dataset is:', test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "faa12be0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-10-28T13:31:59.738914Z",
     "iopub.status.busy": "2023-10-28T13:31:59.737877Z",
     "iopub.status.idle": "2023-10-28T13:31:59.977517Z",
     "shell.execute_reply": "2023-10-28T13:31:59.976125Z"
    },
    "papermill": {
     "duration": 0.268408,
     "end_time": "2023-10-28T13:31:59.981067",
     "exception": false,
     "start_time": "2023-10-28T13:31:59.712659",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e342d8c3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-10-28T13:32:00.036998Z",
     "iopub.status.busy": "2023-10-28T13:32:00.035403Z",
     "iopub.status.idle": "2023-10-28T13:32:00.072241Z",
     "shell.execute_reply": "2023-10-28T13:32:00.070568Z"
    },
    "papermill": {
     "duration": 0.068476,
     "end_time": "2023-10-28T13:32:00.075161",
     "exception": false,
     "start_time": "2023-10-28T13:32:00.006685",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba72cd22",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-10-28T13:32:00.126306Z",
     "iopub.status.busy": "2023-10-28T13:32:00.125810Z",
     "iopub.status.idle": "2023-10-28T13:32:00.290808Z",
     "shell.execute_reply": "2023-10-28T13:32:00.289459Z"
    },
    "papermill": {
     "duration": 0.19403,
     "end_time": "2023-10-28T13:32:00.293672",
     "exception": false,
     "start_time": "2023-10-28T13:32:00.099642",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import other supporting datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "holidays =  pd.read_csv('holidays_events.csv')\n",
    "stores =  pd.read_csv('stores.csv')\n",
    "transactions = pd.read_csv('transactions.csv')\n",
    "oil = pd.read_csv('oil.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "holidays.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "stores.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "transactions.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "oil.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Join the different datasets that have a common feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add oil price\n",
    "train = train.merge(oil, on='date', how='left')\n",
    "test = test.merge(oil, on='date', how='left')\n",
    "\n",
    "# Add transactions\n",
    "train = train.merge(transactions, on=['date', 'store_nbr'], how='left')\n",
    "test = test.merge(transactions, on=['date', 'store_nbr'], how='left')\n",
    "\n",
    "# Add stores description\n",
    "train = train.merge(stores, on='store_nbr', how='left')\n",
    "test = test.merge(stores, on='store_nbr', how='left')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Before adding holidays, apply transformations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Delete transferred holidays\n",
    "holidays = holidays.loc[holidays.iloc[:, -1] != \"True\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add holidays\n",
    "train = train.merge(holidays, on='date', how='left')\n",
    "test = test.merge(holidays, on='date', how='left')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf79e85e",
   "metadata": {
    "papermill": {
     "duration": 0.113021,
     "end_time": "2023-10-28T13:32:00.481709",
     "exception": false,
     "start_time": "2023-10-28T13:32:00.368688",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<a id='goto4'></a>\n",
    "# <h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Data Exploration</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back](#goto0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Adapt holiday depending on whether it is local, regional, national or not a holiday"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We do some transformations on the holiday data to create columns indicating local, regional, and national holidays."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def nat_holiday(row):\n",
    "    if row['locale'] == 'National':\n",
    "        return 1\n",
    "    else:\n",
    "        return 0\n",
    "def reg_holiday(row):\n",
    "    if row['locale'] == 'Regional' and row['locale_name'] == row['state']:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0\n",
    "def loc_holiday(row):\n",
    "    if row['locale'] == 'Local' and row['locale_name'] == row['city']:\n",
    "        return 1\n",
    "    else:\n",
    "        return 0\n",
    "\n",
    "train['holiday_national'] = train.apply(nat_holiday, axis=1)\n",
    "train['holiday_regional'] = train.apply(reg_holiday, axis=1)\n",
    "train['holiday_local'] = train.apply(loc_holiday, axis=1)\n",
    "\n",
    "test['holiday_national'] = test.apply(nat_holiday, axis=1)\n",
    "test['holiday_regional'] = test.apply(reg_holiday, axis=1)\n",
    "test['holiday_local'] = test.apply(loc_holiday, axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Study the characteristics of the datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Number of observations: {test.shape[0]}\\n Number of features: {train.shape[1]}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Time period covered by the data: {train.date.nunique()} days\\n First day: {train.date[0]} || Last day: {train.date.iloc[-1]}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Numbers of stores: {train.store_nbr.nunique()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Convert 'date' to pd.datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['date'] = pd.to_datetime(train['date'], format='%Y-%m-%d')\n",
    "test['date'] = pd.to_datetime(test['date'], format='%Y-%m-%d')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Add day of the week"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['day_of_week'] = train['date'].dt.day_name()\n",
    "test['day_of_week'] = test['date'].dt.day_name()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Eliminate features that do not provide important data\n",
    "\n",
    "* **locale, locale_name, description**: information within holidays\n",
    "* **transferred**: not relevant\n",
    "* **city, state**: information within clusters and type_x\n",
    "* **transactions**: general information that does not separate into products\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = train.drop(columns=['city', 'state', 'transactions', 'type_y', 'locale', 'locale_name', 'description', 'transferred'])\n",
    "test = test.drop(columns=['city', 'state', 'transactions', 'type_y', 'locale', 'locale_name', 'description', 'transferred'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Study behavior of the target series"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(train.groupby('date')['sales'].sum())\n",
    "plt.title('Averages Sales by year')\n",
    "plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%y-%m'))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Group the data by the day of the month and calculate the average sales for each day.\n",
    "average_sales_per_day = train.groupby(train['date'].dt.day)['sales'].mean()\n",
    "\n",
    "# Create a line or bar plot to represent the average sales per day of the month.\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(average_sales_per_day.index, average_sales_per_day.values, marker='o', linestyle='-')\n",
    "plt.xlabel('Day of the Month')\n",
    "plt.ylabel('Average Sales')\n",
    "plt.title('Average Sales per Day of the Month')\n",
    "plt.grid(True)\n",
    "\n",
    "# Show the plot\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.plot(train.groupby('day_of_week')['sales'].sum())\n",
    "plt.title('Averages Sales by day of week')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### First Conclusions:\n",
    "* On the first day of the years there are many no sales\n",
    "* The trend is increasing\n",
    "* Half of each month and year sales increase a lot\n",
    "* Saturdays and Sundays are when sales increase the most"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Much more data can be obtained by doing more preliminary analysis on the data, but our premise in this study is prediction.*"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ae5083d",
   "metadata": {
    "papermill": {
     "duration": 0.037175,
     "end_time": "2023-10-28T13:33:24.426721",
     "exception": false,
     "start_time": "2023-10-28T13:33:24.389546",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "<a id='goto5'></a>\n",
    "# <h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Data Modeling</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back](#goto0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto5a'></a>\n",
    "# <h2 style=\"background-color:gold;font-family:newtimeroman;color:black;font-size:200%;text-align:center;border-radius: 50px 15px;\">Preprocessing</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back to models](#goto5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Clean NaN observations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(train.isna().sum().sort_values(ascending=False) / train.shape[0] * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use the next lowest value first and then the next highest value\n",
    "train['dcoilwtico'] = train['dcoilwtico'].fillna(method='bfill')\n",
    "train['dcoilwtico'] = train['dcoilwtico'].fillna(method='ffill')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.dcoilwtico.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(test.isna().sum().sort_values(ascending=False) / test.shape[0] * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use the next lowest value first and then the next highest value\n",
    "test['dcoilwtico'] = test['dcoilwtico'].fillna(method='bfill')\n",
    "test['dcoilwtico'] = test['dcoilwtico'].fillna(method='ffill')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.dcoilwtico.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create Dataset for each store and product"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "\n",
    "train_dict = {}\n",
    "\n",
    "for store in train['store_nbr'].unique():\n",
    "    for product in train['family'].unique():\n",
    "\n",
    "        subset_df = train[(train['store_nbr'] == store) & (train['family'] == product)]\n",
    "\n",
    "        key = f'train_{store}_{product}'.replace('/', '_').replace(' ', '_')\n",
    "\n",
    "        train_dict[key] = subset_df\n",
    "\n",
    "test_dict = {}\n",
    "\n",
    "for store in test['store_nbr'].unique():\n",
    "    for product in test['family'].unique():\n",
    "\n",
    "        subset_df = test[(test['store_nbr'] == store) & (test['family'] == product)]\n",
    "\n",
    "        key = f'test_{store}_{product}'.replace('/', '_').replace(' ', '_')\n",
    "\n",
    "        test_dict[key] = subset_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Save Dataframe splits in small sets\n",
    "# import os\n",
    "\n",
    "\n",
    "# if not os.path.exists('keys'):\n",
    "#     os.makedirs('keys')\n",
    "\n",
    "# for key in train_dict.keys():\n",
    "#     train_dict[key].to_csv(f'keys/{key}.csv', index=False)\n",
    "# for key in test_dict.keys():\n",
    "#     test_dict[key].to_csv(f'keys/{key}.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto5b'></a>\n",
    "# <h2 style=\"background-color:gold;font-family:newtimeroman;color:black;font-size:200%;text-align:center;border-radius: 50px 15px;\">Prophet Model</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back to models](#goto5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Use prophet to train the temporal model and make prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from prophet import Prophet\n",
    "from prophet.serialize import model_to_json, model_from_json"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Prophet model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_prophet_model(\n",
    "            df: pd.DataFrame, \n",
    "            target: str, \n",
    "            dates: str, \n",
    "            ):\n",
    "        \"\"\"\n",
    "        Train and fit a Prophet model for time series forecasting.\n",
    "\n",
    "        Parameters:\n",
    "            df (pd.DataFrame): The DataFrame containing the time series data.\n",
    "            target (str): The name of the column containing the target values.\n",
    "            dates (str): The name of the column containing the corresponding dates.\n",
    "\n",
    "        Returns:\n",
    "            Prophet: The fitted Prophet model.\n",
    "\n",
    "        Example:\n",
    "            # Train the Prophet model\n",
    "            best_model = ProphetModel.train_prophet_model(df, 'value', 'date')\n",
    "\n",
    "            # Make predictions with the model\n",
    "            future = best_model.make_future_dataframe(periods=10)\n",
    "            forecast = best_model.predict(future)\n",
    "\n",
    "            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))\n",
    "        \"\"\"\n",
    "        \n",
    "        # Prepare the data in the format required by Prophet\n",
    "        df_prophet = df.rename(columns={target: 'y', dates: 'ds'})\n",
    "\n",
    "        best_model = Prophet(changepoint_prior_scale = 3.5).fit(df_prophet)\n",
    "\n",
    "        return best_model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train splits little models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "\n",
    "for key in train_dict.keys():\n",
    "    train_dict[key] = pd.get_dummies(train_dict[key], drop_first=True)\n",
    "    model = train_prophet_model(train_dict[key], 'sales', 'date')\n",
    "    with open(f'models/{key}.json', 'w') as fout:\n",
    "        fout.write(model_to_json(model))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Predict models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%time\n",
    "\n",
    "ids = []\n",
    "sales = []\n",
    "\n",
    "for key in train_dict.keys():\n",
    "    with open(f'models/{key}.json', 'r') as fin:\n",
    "        model_json = fin.read()\n",
    "        model = model_from_json(model_json)\n",
    "    name = f'test{key[5:]}'\n",
    "    test_dict[name] = pd.get_dummies(test_dict[name], drop_first=True)\n",
    "    test_dict[name].rename(columns = {'date': 'ds'}, inplace=True)\n",
    "    t = test_dict[name].drop(columns=['id'])\n",
    "    predict = model.predict(t)\n",
    "    ids.extend(test_dict[name]['id'])\n",
    "    sales.extend(predict['yhat'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto5c'></a>\n",
    "# <h2 style=\"background-color:gold;font-family:newtimeroman;color:black;font-size:200%;text-align:center;border-radius: 50px 15px;\">Final Touches</h2>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back to models](#goto5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create final Dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission = pd.DataFrame()\n",
    "submission['id'] = ids\n",
    "submission['sales'] = sales"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Convert to csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission.to_csv('output10.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='goto6'></a>\n",
    "# <h1 style=\"background-color:orange;font-family:newtimeroman;color:black;font-size:300%;text-align:center;border-radius: 15px 50px;\">Conclusions</h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "[Back](#goto0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Submissions Scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**output8.csv:** (changepoint_prior_scale = 3)\n",
    "* Score (rmsle): 0.46566\n",
    "\n",
    "**output10.csv:** (changepoint_prior_scale = 4)\n",
    "* Score (rmsle): 0.46577\n",
    "\n",
    "**output7.csv:** (changepoint_prior_scale = 5)\n",
    "* Score (rmsle): 0.46633\n",
    "\n",
    "**output9.csv:** (changepoint_prior_scale = 2)\n",
    "* Score (rmsle): 0.46741\n",
    "\n",
    "**output5.csv:** (changepoint_prior_scale = 1)\n",
    "* Score (rmsle): 0.46778\n",
    "\n",
    "**output5.csv:** (changepoint_prior_scale = 10)\n",
    "* Score (rmsle): 0.47033\n",
    "\n",
    "**output.csv:** (changepoint_prior_scale = 0.5, seasonality_prior_scale = 10)\n",
    "* Score (rmsle): 0.47767\n",
    "\n",
    "**output3.csv:** (changepoint_prior_scale = 0.5, seasonality_prior_scale = 1)\n",
    "* Score (rmsle): 0.47767\n",
    "\n",
    "**output4.csv:** (changepoint_prior_scale = 0.5, holidays_prior_scale = 0.01)\n",
    "* Score (rmsle): 0.47767\n",
    "\n",
    "**submission.csv:** (changepoint_prior_scale = 0.05 (default), seasonality_prior_scale = 10 (default))\n",
    "* Score (rmsle): 0.51021\n",
    "\n",
    "**output1.csv:** (changepoint_prior_scale = 0.1, seasonality_prior_scale = 1)\n",
    "* Score (rmsle): 0.51506\n",
    "\n",
    "**output2.csv:** (changepoint_prior_scale = 0.5, seasonality_prior_scale = 10, seasonality_mode = 'multiplicative')\n",
    "* Score (rmsle): 0.64443"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 10651.432121,
   "end_time": "2023-10-28T16:29:17.646365",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-10-28T13:31:46.214244",
   "version": "2.4.0"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {
     "23209c2d8d10498da39e2e576df1298e": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_76029792948448528513186dbf5bd457",
       "placeholder": "​",
       "style": "IPY_MODEL_5b870f66b23f43ceb939621c8a0ad92c",
       "value": "100%"
      }
     },
     "2b1c981a35ce465ab8d5fa77b7c1e750": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "365421af2e964af6b174e1eac02a1ca3": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "5665337a67d74c04a1f744e3ff0ccfda": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "5b870f66b23f43ceb939621c8a0ad92c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "5dd7395d88dd4014acd55447fe296538": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "62af5d19fe4d474aae4c91cc8a1e258c": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_5dd7395d88dd4014acd55447fe296538",
       "max": 100,
       "min": 0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_fa9ea09ec5c446d79d8d7a9cdbbb36c5",
       "value": 100
      }
     },
     "76029792948448528513186dbf5bd457": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "7d318e0ba9fb4dd095be50171a324618": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_a2652941fb484faaa011c817768c70d7",
       "placeholder": "​",
       "style": "IPY_MODEL_5665337a67d74c04a1f744e3ff0ccfda",
       "value": "100%"
      }
     },
     "863287f88f674a4480c34808f35bed70": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "a2652941fb484faaa011c817768c70d7": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "a3202d2a65c64256a05402238e7b953a": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_7d318e0ba9fb4dd095be50171a324618",
        "IPY_MODEL_bf8decc34c654ce38f1bdfc348b8b0e1",
        "IPY_MODEL_e611a23ad342414bb0e7a67169bfa431"
       ],
       "layout": "IPY_MODEL_365421af2e964af6b174e1eac02a1ca3"
      }
     },
     "abd950d6eb9d4f5ca811be585e6c0cb0": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     },
     "ad861ace6d58411283adcced3902eec9": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "DescriptionStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "DescriptionStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "description_width": ""
      }
     },
     "bc63ff00d467428e9894ee19fdcfe4bf": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_c917bd42cf22461ab3035c72b5db6945",
       "placeholder": "​",
       "style": "IPY_MODEL_ad861ace6d58411283adcced3902eec9",
       "value": " 100/100 [03:51&lt;00:00,  2.40s/it]"
      }
     },
     "bf8decc34c654ce38f1bdfc348b8b0e1": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "FloatProgressModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "FloatProgressModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "ProgressView",
       "bar_style": "success",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_e9672958e24948b5acb245d51baf73ce",
       "max": 10000,
       "min": 0,
       "orientation": "horizontal",
       "style": "IPY_MODEL_abd950d6eb9d4f5ca811be585e6c0cb0",
       "value": 10000
      }
     },
     "c917bd42cf22461ab3035c72b5db6945": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "caa58b167b4b45a38a1d4620a268b671": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HBoxModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HBoxModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HBoxView",
       "box_style": "",
       "children": [
        "IPY_MODEL_23209c2d8d10498da39e2e576df1298e",
        "IPY_MODEL_62af5d19fe4d474aae4c91cc8a1e258c",
        "IPY_MODEL_bc63ff00d467428e9894ee19fdcfe4bf"
       ],
       "layout": "IPY_MODEL_863287f88f674a4480c34808f35bed70"
      }
     },
     "cf965cbccaea4e7382ec9f06086400cf": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "e611a23ad342414bb0e7a67169bfa431": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "HTMLModel",
      "state": {
       "_dom_classes": [],
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "HTMLModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/controls",
       "_view_module_version": "1.5.0",
       "_view_name": "HTMLView",
       "description": "",
       "description_tooltip": null,
       "layout": "IPY_MODEL_cf965cbccaea4e7382ec9f06086400cf",
       "placeholder": "​",
       "style": "IPY_MODEL_2b1c981a35ce465ab8d5fa77b7c1e750",
       "value": " 10000/10000 [11:46&lt;00:00, 13.85it/s]"
      }
     },
     "e9672958e24948b5acb245d51baf73ce": {
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "model_name": "LayoutModel",
      "state": {
       "_model_module": "@jupyter-widgets/base",
       "_model_module_version": "1.2.0",
       "_model_name": "LayoutModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "LayoutView",
       "align_content": null,
       "align_items": null,
       "align_self": null,
       "border": null,
       "bottom": null,
       "display": null,
       "flex": null,
       "flex_flow": null,
       "grid_area": null,
       "grid_auto_columns": null,
       "grid_auto_flow": null,
       "grid_auto_rows": null,
       "grid_column": null,
       "grid_gap": null,
       "grid_row": null,
       "grid_template_areas": null,
       "grid_template_columns": null,
       "grid_template_rows": null,
       "height": null,
       "justify_content": null,
       "justify_items": null,
       "left": null,
       "margin": null,
       "max_height": null,
       "max_width": null,
       "min_height": null,
       "min_width": null,
       "object_fit": null,
       "object_position": null,
       "order": null,
       "overflow": null,
       "overflow_x": null,
       "overflow_y": null,
       "padding": null,
       "right": null,
       "top": null,
       "visibility": null,
       "width": null
      }
     },
     "fa9ea09ec5c446d79d8d7a9cdbbb36c5": {
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "model_name": "ProgressStyleModel",
      "state": {
       "_model_module": "@jupyter-widgets/controls",
       "_model_module_version": "1.5.0",
       "_model_name": "ProgressStyleModel",
       "_view_count": null,
       "_view_module": "@jupyter-widgets/base",
       "_view_module_version": "1.2.0",
       "_view_name": "StyleView",
       "bar_color": null,
       "description_width": ""
      }
     }
    },
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
