#!/usr/bin/env python
# coding: utf-8

# # Chapter 7:  Data Science

# ## Ingest

# *COVID-19 Data from [New York Times Github](https://github.com/nytimes/covid-19-data)*

# In[1]:


import pandas as pd
df = pd.read_csv("covid19.csv")
#df.to_csv("covid19.csv", index=False)
df.head()


# Last five rows

# In[2]:


df.tail()


# *What are the columns?*

# In[3]:


df.columns


# *What is the shape:  i.e. rows,columns?*

# In[4]:


df.shape


# ## EDA

# *What are general characteristics of the data?  A good way to find out is `df.describe`*

# In[5]:


df.describe()


# *Cases and Deaths in the USA due to Covid-19*

# In[6]:


import seaborn as sns
sns.scatterplot(x="cases", y="deaths", 
                hue="deaths",size="deaths", data=df)


# ### Date-based EDA

# *Dealing with dates by setting index*

# In[7]:


df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
print(f"Columns: {df.columns}")
df.index


# *Filter results by date range*

# In[8]:


from datetime import date, timedelta
last_date = df.index.max()
today = last_date
daybefore = today - timedelta(days = 2)
print(f"Today {today}")
print(f"Two days ago {daybefore}") 


# In[9]:


df.head()


# In[10]:


df.loc[daybefore:today].head()


# *The distribution of the data by date*

# In[11]:


sns.kdeplot(df.loc[daybefore:today]["deaths"], shade=True)


# *Sort DataFrame in place by states with highest deaths and cases and show first 10 results*

# In[12]:


current_df = df.loc[daybefore:today].sort_values(by=["deaths", "cases"], ascending=False)
current_df.head(10)


# *There should be 50 states and District of Columbia*

# In[13]:


current_df.shape


# ### State Based Analysis

# *Get 10 states and subset*

# In[14]:


top_ten_states = list(current_df["state"].head(10).values)
top_ten_states


# In[15]:


top_states_df = df[df['state'].isin(top_ten_states)].drop(columns="fips")


# *Verify the unique states left is the same 10*

# In[16]:


set(top_states_df.state.values)


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
ax = sns.lmplot(x="cases", y="deaths",
                    height=3.5,
                    col="state",
                    col_wrap=5,
                    hue="state", 
                    palette="Set2",
                    data=top_states_df)
ax.fig.subplots_adjust(wspace=.2)


# *Interactive plot of top states*

# In[18]:


top_states_march_current_df = top_states_df.loc["2020-03-08":today].sort_values(by=["deaths", "cases"], ascending=True)
top_states_march_current_df.head()


# ### Search for Features:  Political, Health and Finance

# *Sugar Intake By State*

# In[19]:


cdc_2013 = pd.read_csv("education_sugar_cdc_2003.csv")
cdc_2013.to_csv("education_sugar_cdc_2003.csv", index=False)
cdc_2013.set_index("State", inplace=True)
for column in cdc_2013.columns:
  cdc_2013[column]=cdc_2013[column].str.replace(r"\(.*\)","")
  cdc_2013[column]=pd.to_numeric(cdc_2013[column])
  
cdc_2013.reset_index(inplace=True)
cdc_2013.rename(columns={"State": "state", "Employed": "employed-sugar-intake-daily"},inplace=True)
cdc_2013.head()


# *Combine Sugar Data and Covid-19 Data*

# In[20]:


cdc_employed_df = cdc_2013[["employed-sugar-intake-daily", "state"]]
sugar_covid_df = df.merge(cdc_employed_df, how="inner", on="state")
sugar_covid_df.head()


# *What about data from the 2016 Election?*

# In[21]:


election_df = pd.read_csv("2016-Electoral-Votes.csv")
#election_df.to_csv("2016-Electoral-Votes.csv", index=False)
election_df.rename(columns={"State": "state"},inplace=True)
election_df.drop(columns="Votes", inplace=True)
election_df = pd.concat([election_df, pd.get_dummies(election_df["Winning Party"])], axis=1);
election_df.head()
#election_df["Republican?"] = pd.get_dummies(election_df, columns=["Winning Party"])
#election_df.head()


# In[22]:


sugar_covid_df = sugar_covid_df.merge(election_df, how="inner", on="state")
sugar_covid_df.head()


# *Generate Heatmap*

# In[23]:


sugar_covid_df.corr()


# In[24]:


sugar_covid_df.to_csv("covid-eda.csv")


# 

# ## Modeling

# 

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assuming 'date' is our predictor variable
df_covid = sugar_covid_df
X = pd.to_numeric(pd.to_datetime(df_covid.index)).values.reshape(-1, 1)
y = df_covid['cases'].values

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[26]:


# Visualizing the model's predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression Model: Actual vs Predicted COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# ## Conclusion
# 

# 

# In[ ]:


##We explored the COVID-19 dataset from the New York Times GitHub repository.
Found that the dataset contains information on COVID-19 cases and deaths across different states in the USA.
Conducted exploratory data analysis (EDA) to understand the dataset's general characteristics, trends over time, and state-based patterns.
Discovered that COVID-19 cases and deaths have varied significantly over time and across different states, with some states experiencing higher numbers than others.
Integrated external datasets like sugar intake and 2016 election data to explore potential correlations with COVID-19 outcomes.
Identified correlations between certain external factors and COVID-19 outcomes, suggesting potential areas for further investigation.
Attempted to predict COVID-19 cases using linear regression and found that while the model showed some predictive capability, further refinement and feature engineering may be necessary to improve accuracy.
Overall, this analysis provides valuable insights into the dynamics of the COVID-19 pandemic and its potential associations with external factors, aiding in our understanding and response to the ongoing crisis.

