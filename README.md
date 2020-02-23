# Student Project #2
## Northwind Database

For this project, you'll be working with the Northwind database--a free, open-source dataset created by Microsoft containing data from a fictional company. You probably remember the Northwind database from our section on Advanced SQL.

The goal of this project is to test your ability to gather information from a real-world database and use your knowledge of statistical analysis and hypothesis testing to generate analytical insights that can be of value to the company. 

<br /><br />
<img src="Northwind_ERD.png" style="width:80%;" alt="Database Schema" />
<br /><br />

# The Goal

The goal of your project is to query the database to get the data needed to perform a statistical analysis. In this statistical analysis, we'll need to perform a hypothesis test (or perhaps several) to answer the following questions:

- **Question 1**: Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount?
- **Question 2**: Is there a statistically significant difference in performance between employees from US and UK?
- **Question 3**: Is there statistically significant difference in discounts given by USA and UK employees?
- **Question 4**: Is there a statistically significant difference in demand of produce each month?
- **Question 5**: Is there a statistically significant difference in discount between categories?
- **Question 6**: Is there a statistically significant difference in performance of shipping companies?

## Importing libraries


```python
import sqlite3 # for database
import pandas as pd # for dataframe
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import numpy as np # analysis
from scipy import stats # significance levels, normality
import itertools # for combinations
import statsmodels.api as sm # anova
from statsmodels.formula.api import ols

import warnings
warnings.filterwarnings('ignore') # hide matplotlib warnings
```

## Connecting to database


```python
# Connecting to database
conn = sqlite3.connect('Northwind_small.sqlite')
c = conn.cursor()
```


```python
# List of all tables
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
tables = [i[0] for i in tables]
```

## Converting all tables into dataframes


```python
# Loop to put all tables into pandas dataframes
dfs = []
for i in tables:
    table = c.execute('select * from "'+i+'"').fetchall()
    columns = c.execute('PRAGMA table_info("'+i+'")').fetchall()
    df = pd.DataFrame(table, columns=[i[1] for i in columns])
    # Cute little function to make a string into variable name
    foo = i+"_df"
    exec(foo + " = df") # => TableName_df
    # Keep all dataframe names in the list to remember what we have
    dfs.append(foo)
```

# Exploratory Data Analysis

## Order quantities of disconted and not discounted products

Table `Product` has 77 entries, each entry is unique product

First we can check visually if discount really made a difference in order quantity


```python
discount = OrderDetail_df[OrderDetail_df['Discount']!=0].groupby('ProductId')['Quantity'].mean()
no_discount = OrderDetail_df[OrderDetail_df['Discount']==0].groupby('ProductId')['Quantity'].mean()
plt.figure(figsize=(16,5))
plt.bar(discount.index, discount.values, alpha=1, label='Discount', color='#a0b0f0')
plt.bar(no_discount.index, no_discount.values, alpha=0.8, label='No Discount', color='#c9f9a0')
plt.legend()
plt.title('Order quantities with/without discount')
plt.xlabel('Product ID')
plt.ylabel('Average quantities')
plt.show()

print('Conclusion')
print("On average {}% of discounted products were sold in larger quantities".format(round(sum(discount.values > no_discount.values)/len(discount.values)*100),2))
print("Average order quantity with discount - {} items, without - {} items".format(round(discount.values.mean(),2), round(no_discount.values.mean(),2)))
```


![png](images/Project%202%20Student_16_0.png)


    Conclusion
    On average 70.0% of discounted products were sold in larger quantities
    Average order quantity with discount - 26.43 items, without - 21.81 items


There is evidence that customers tend to buy more product if it was discounted

To prove that our hypothesis correct we will run an experiment

## Orders grouped by discount level

First lets see how many discounts we have and how many orders on average each discont level provides


```python
# Let's get all discount levels
discounts = OrderDetail_df['Discount'].unique()
discounts.sort()
print('Discount levels')
print(discounts)
```

    Discount levels
    [0.   0.01 0.02 0.03 0.04 0.05 0.06 0.1  0.15 0.2  0.25]



```python
# Group orders by discount amounts
# Each group is a DataFrame containing orders with certain discount level
groups = {}
for i in discounts:
    groups[i] = OrderDetail_df[OrderDetail_df['Discount']==i]
```


```python
# Create new DataFrame with Discounts and Order quantities
discounts_df = pd.DataFrame(columns=['Discount %','Orders','Avg. Order Quantity'])
for i in groups.keys():
    discounts_df = discounts_df.append({'Discount %':i*100,'Orders':len(groups[i]),'Avg. Order Quantity':groups[i]['Quantity'].mean()}, ignore_index=True)

discounts_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Discount %</th>
      <th>Orders</th>
      <th>Avg. Order Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1317.0</td>
      <td>21.715262</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>185.0</td>
      <td>28.010811</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10.0</td>
      <td>173.0</td>
      <td>25.236994</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15.0</td>
      <td>157.0</td>
      <td>28.382166</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20.0</td>
      <td>161.0</td>
      <td>27.024845</td>
    </tr>
    <tr>
      <th>10</th>
      <td>25.0</td>
      <td>154.0</td>
      <td>28.240260</td>
    </tr>
  </tbody>
</table>
</div>



Table above shows that 1%, 2%, 3%, 4% and 6% discounts are significantly small and to draw a conclusion would be problematic.
Besides that customers ordered such a small quantities it looks like it made a negative impact. Probably because it was a promotion of new product or some other reason.

Lets drop these discount levels from our experiment

## Bootstrap

Bootstrapping is a type of resampling where large numbers of smaller samples of the same size are repeatedly drawn, with replacement, from a single original sample.


```python
def bootstrap(sample, n):
    bootstrap_sampling_dist = []
    for i in range(n):
        bootstrap_sampling_dist.append(np.random.choice(sample, size=len(sample), replace=True).mean())
    return np.array(bootstrap_sampling_dist)
```

## Cohen's d

Cohen's d is an effect size used to indicate the standardised difference between two means. It can be used, for example, to accompany reporting of t-test and ANOVA results. It is also widely used in meta-analysis. Cohen's d is an appropriate effect size for the comparison between two means.


```python
def Cohen_d(group1, group2):

    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()
    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    return abs(d)
```

## Visualization


```python
def visualization(control, experimental):
    plt.figure(figsize=(10,6))
    sns.distplot(experimental, bins=50,  label='Experimental')
    sns.distplot(control, bins=50,  label='Control')

    plt.axvline(x=control.mean(), color='k', linestyle='--')
    plt.axvline(x=experimental.mean(), color='k', linestyle='--')

    plt.title('Control and Experimental Sampling Distributions', fontsize=14)
    plt.xlabel('Distributions')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
```

# Question #1

## Does discount amount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount?

- $H_0$: there is no difference in order quantity due to discount
- $H_\alpha$: there is an increase in order quantity due to discount

Usually discount increases order quantity, so it would be reasonable to perform one-tailed test with $\alpha$ set to 0.025. If $p$ < $\alpha$, we reject null hypothesis.

## Welch's T-test

In statistics, Welch's t-test, or unequal variances t-test, is a two-sample location test which is used to test the hypothesis that two populations have equal means.

At first I created two distributions (control and experimental). Control distribution uncludes only order quantities without discount only, and experimental distribution includes order quantities with discount (at any level)

This experiment would answer a question if there is any difference in purchase quantity


```python
control = OrderDetail_df[OrderDetail_df['Discount']==0]['Quantity']
experimental = OrderDetail_df[OrderDetail_df['Discount']!=0]['Quantity']

t_stat, p = stats.ttest_ind(control, experimental)
d = Cohen_d(experimental, control)

print('Reject Null Hypothesis') if p < 0.025 else print('Failed to reject Null Hypothesis')
print("Cohen's d:", d)
visualization(control, experimental)
```

    Reject Null Hypothesis
    Cohen's d: 0.2862724481729283



![png](images/Project%202%20Student_37_1.png)


Result of the experiment shows that there is a _**statistically significant**_ difference in orders quantities, hence we reject null hypothesis

The question is posed in the way that it is asking if order quantity is different at the different discount level.

The following step in the research would be to answer the question about at what discount level we see _**statisticaly significant**_ difference in orders quantities

We will follow the same process as previous experiment, but this time we'll break our experimental group into discount amounts


```python
discounts_significance_df = pd.DataFrame(columns=['Discount %','Null Hypothesis','Cohens d'], index=None)

discounts = [0.05, 0.1, 0.15, 0.2, 0.25]
control = OrderDetail_df[OrderDetail_df['Discount']==0]['Quantity']
for i in discounts:
    experimental = OrderDetail_df[OrderDetail_df['Discount']==i]['Quantity']
    st, p = stats.ttest_ind(control, experimental)
    d = Cohen_d(experimental, control)
    discounts_significance_df = discounts_significance_df.append( { 'Discount %' : str(i*100)+'%' , 'Null Hypothesis' : 'Reject' if p < 0.025 else 'Failed', 'Cohens d' : d } , ignore_index=True)    

discounts_significance_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Discount %</th>
      <th>Null Hypothesis</th>
      <th>Cohens d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0%</td>
      <td>Reject</td>
      <td>0.346877</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0%</td>
      <td>Reject</td>
      <td>0.195942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0%</td>
      <td>Reject</td>
      <td>0.372404</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0%</td>
      <td>Reject</td>
      <td>0.300712</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25.0%</td>
      <td>Reject</td>
      <td>0.366593</td>
    </tr>
  </tbody>
</table>
</div>



Result of the test shows that there is _**statistically significant**_ difference in quantities between orders with no discount and applied discounts of 5%, 10%, 15%, 20%, 25%. Hence we reject null hypothesis

## Statistically significant difference between discount levels

- $H_0$: there is no difference in order quantity between discounts
- $H_\alpha$: there is a difference in order quantity between discounts


```python
discounts = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
comb = itertools.combinations(discounts, 2)
discount_levels_df = pd.DataFrame(columns=['Discount %','Null Hypothesis','Cohens d'], index=None)

for i in comb:
    
    control =      OrderDetail_df[OrderDetail_df['Discount']==i[0]]['Quantity']
    experimental = OrderDetail_df[OrderDetail_df['Discount']==i[1]]['Quantity']
    
    st, p = stats.ttest_ind(experimental, control)
    d = Cohen_d(experimental, control)
    
    discount_levels_df = discount_levels_df.append( { 'Discount %' : str(i[0]*100)+'% - '+str(i[1]*100)+'%', 'Null Hypothesis' : 'Reject' if p < 0.05 else 'Failed', 'Cohens d' : d } , ignore_index=True)    

discount_levels_df.sort_values('Cohens d', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Discount %</th>
      <th>Null Hypothesis</th>
      <th>Cohens d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>10.0% - 15.0%</td>
      <td>Failed</td>
      <td>0.149332</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.0% - 25.0%</td>
      <td>Failed</td>
      <td>0.145146</td>
    </tr>
    <tr>
      <th>0</th>
      <td>5.0% - 10.0%</td>
      <td>Failed</td>
      <td>0.127769</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10.0% - 20.0%</td>
      <td>Failed</td>
      <td>0.089008</td>
    </tr>
    <tr>
      <th>7</th>
      <td>15.0% - 20.0%</td>
      <td>Failed</td>
      <td>0.068234</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20.0% - 25.0%</td>
      <td>Failed</td>
      <td>0.062415</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0% - 20.0%</td>
      <td>Failed</td>
      <td>0.047644</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0% - 15.0%</td>
      <td>Failed</td>
      <td>0.017179</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0% - 25.0%</td>
      <td>Failed</td>
      <td>0.010786</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15.0% - 25.0%</td>
      <td>Failed</td>
      <td>0.006912</td>
    </tr>
  </tbody>
</table>
</div>



Result of the test shows that there is no _**statistically significant**_ difference in order quantity between discounts of 5%, 10%, 15%, 20% and 25%.

# Question 2
## Is there a statistically significant difference in performance between employees from US and UK?

- $H_0$: There is no difference in performance between US and UK employees
- $H_\alpha$: there is a difference in performance between US and UK employees

How to measure performance of employee?
It could be done in different ways, such a:
- survey of the customers
- amount of orders they were able to process
- time it took them to procees the orders
- etc...

To find out the difference in performance we will perform two tests


```python
employees_orders = pd.read_sql_query( '''
                                    
                                SELECT O.EmployeeId, E.Country, COUNT(O.Id) AS Total_Orders  
                                FROM [Order] AS O
                                JOIN Employee as E
                                ON O.EmployeeId = E.Id
                                GROUP BY O.EmployeeId
                                
                                ''' ,conn)
```


```python
employees_orders
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmployeeId</th>
      <th>Country</th>
      <th>Total_Orders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>USA</td>
      <td>123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>USA</td>
      <td>96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>USA</td>
      <td>127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>USA</td>
      <td>156</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>UK</td>
      <td>42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>UK</td>
      <td>67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>UK</td>
      <td>72</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>USA</td>
      <td>104</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>UK</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



Even without significance test we can tell there is a big difference in the amount of total orders two groups were able to process in two years.

## 2.1 Amount of orders processed by US and UK employees


```python
# ANOVA Test
formula = 'Total_Orders ~ C(Country)'
lm = ols(formula, employees_orders).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                     sum_sq   df          F    PR(>F)
    C(Country)  9446.755556  1.0  22.640129  0.002064
    Residual    2920.800000  7.0        NaN       NaN


Result of ANOVA Test shows that there is _**statistically significant**_ difference in orders quantity between two groups of employees from USA and UK.

My suspicion was that group from USA covers larger territory, but it's turn out not to be the case.

I want to investigate further into performance of employees and compare their order processing time, maybe thats the reason in such a big difference in amount of orders

## 2.2 Order Processing Time by US vs UK Employees

In this test I want to figure out if number of orders affected by how fast employees can process them


```python
usa_uk = pd.read_sql_query('''
                    
                    SELECT O.Id, O.OrderDate, O.ShippedDate, E.Country FROM [Order] AS O
                    JOIN Employee AS E
                    ON O.EmployeeId = E.Id

''',conn)
```


```python
usa_uk.OrderDate = pd.to_datetime(usa_uk.OrderDate)
usa_uk.ShippedDate = pd.to_datetime(usa_uk.ShippedDate)
usa_uk['ProcessingTime'] = usa_uk.ShippedDate - usa_uk.OrderDate
usa_uk.ProcessingTime = usa_uk.ProcessingTime.dt.days
```


```python
usa_uk.dropna(inplace=True)
```


```python
usa = usa_uk[usa_uk.Country == 'USA']['ProcessingTime']
uk  = usa_uk[usa_uk.Country == 'UK']['ProcessingTime']

print(usa.mean(), uk.mean())
stats.ttest_ind(usa, uk)
print(Cohen_d(usa, uk))
```

    8.375634517766498 8.807339449541285
    0.06310985453186797


Result of the test shows that there is no _**statistically significant**_ difference in processing time, hence we  falied to reject null hypothesis

# Question 3
## Is there statistically significant difference in discounts given by USA and UK employees?

- $H_0$: There is no difference in discounts given by from USA and UK employees
- $H_\alpha$: There is a difference in discounts given by from USA and UK employees

### Read Database


```python
usa_uk_discount = pd.read_sql_query('''

                    SELECT OD.Discount, E.Country FROM [Order] AS O
                    JOIN OrderDetail AS OD ON O.Id = OD.OrderId
                    JOIN Employee AS E ON O.EmployeeId = E.Id

''', conn)
```


```python
formula = 'Discount ~ C(Country)'
lm = ols(formula, usa_uk_discount).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                   sum_sq      df         F    PR(>F)
    C(Country)   0.067081     1.0  9.671415  0.001896
    Residual    14.933259  2153.0       NaN       NaN


### Result

Result of the test shows that there is _**statistically significant**_ difference in discount amount between employees from USA and UK, hence we reject null hypothesis

Employees from USA tend to give smaller discount to their clients

# Question 4
## Is there a statistically significant difference in demand of produce each month?

- $H_0$: There is no difference in demand of produce each month
- $H_\alpha$: There is a difference in demand of produce each month

### Read Database


```python
produce = pd.read_sql_query('''

                                SELECT O.OrderDate, OD.Quantity, OD.Discount, CategoryId FROM [Order] AS O
                                JOIN OrderDetail AS OD
                                ON O.Id = OD.OrderId
                                JOIN Product
                                ON Product.Id = OD.ProductId
                                WHERE Product.CategoryId = 7

''',conn)   
```

### Group data by month


```python
produce.OrderDate = pd.to_datetime(produce.OrderDate)
produce['Month'] = produce.OrderDate.dt.month
```


```python
produce.groupby('Month').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>CategoryId</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>16.545455</td>
      <td>0.050000</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.555556</td>
      <td>0.011111</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.500000</td>
      <td>0.004545</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29.105263</td>
      <td>0.028947</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.888889</td>
      <td>0.075556</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21.285714</td>
      <td>0.085714</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26.375000</td>
      <td>0.050000</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>15.666667</td>
      <td>0.038889</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17.500000</td>
      <td>0.025000</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>33.250000</td>
      <td>0.037500</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>16.000000</td>
      <td>0.055556</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>26.842105</td>
      <td>0.100000</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



### ANOVA Test


```python
formula = 'Quantity ~ C(Month)'
lm = ols(formula, produce).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                    sum_sq     df         F    PR(>F)
    C(Month)   4834.012843   11.0  1.318794  0.221691
    Residual  41319.957745  124.0       NaN       NaN


### Result

There is no _**statistically significant**_ difference in order quantity between months, hence we failed to reject null hypothesis

# Question 5
## Is there a statistically significant difference in discount between categories?

- $H_0$: There is no difference in discount level between categories
- $H_\alpha$: There is a difference in discount level between categories

### Read Database


```python
category_discount = pd.read_sql_query('''

                        SELECT OrderDetail.UnitPrice, Discount, CategoryId FROM OrderDetail
                        JOIN Product
                        ON OrderDetail.ProductId = Product.Id

''',conn)
```

### ANOVA Test


```python
formula = 'Discount ~ C(CategoryId)'
lm = ols(formula, category_discount).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                      sum_sq      df         F    PR(>F)
    C(CategoryId)   0.074918     7.0  1.539545  0.149326
    Residual       14.925422  2147.0       NaN       NaN


### Result

Result of the test shows that there is no _**statistically significant**_ difference in discount level between categories, hence we failed to reject null hypothesis

# Question 6
## Is there a statistically significant difference in performance of shipping companies?

- $H_0$: There is no difference in discount level between categories
- $H_\alpha$: There is a difference in discount level between categories


```python
Order_df.OrderDate = pd.to_datetime(Order_df.OrderDate)
Order_df.ShippedDate = pd.to_datetime(Order_df.ShippedDate)
Order_df.RequiredDate = pd.to_datetime(Order_df.RequiredDate)

Order_df['ProcessingTime'] = Order_df.ShippedDate - Order_df.OrderDate
Order_df['ShippingTime'] = Order_df.RequiredDate - Order_df.ShippedDate

Order_df.ShippingTime = Order_df.ShippingTime.dt.days
Order_df.ProcessingTime = Order_df.ProcessingTime.dt.days
```


```python
Order_df.groupby('ShipVia').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>EmployeeId</th>
      <th>Freight</th>
      <th>ProcessingTime</th>
      <th>ShippingTime</th>
    </tr>
    <tr>
      <th>ShipVia</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>10667.594378</td>
      <td>4.232932</td>
      <td>65.001325</td>
      <td>8.571429</td>
      <td>19.485714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10674.963190</td>
      <td>4.536810</td>
      <td>86.640644</td>
      <td>9.234921</td>
      <td>18.765079</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10641.592157</td>
      <td>4.400000</td>
      <td>80.441216</td>
      <td>7.473896</td>
      <td>19.963855</td>
    </tr>
  </tbody>
</table>
</div>




```python
formula = 'ProcessingTime ~ C(ShipVia)'
lm = ols(formula, Order_df).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```

                      sum_sq     df         F    PR(>F)
    C(ShipVia)    433.501581    2.0  4.676819  0.009563
    Residual    37354.696194  806.0       NaN       NaN



```python
Shipper_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>CompanyName</th>
      <th>Phone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Speedy Express</td>
      <td>(503) 555-9831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>United Package</td>
      <td>(503) 555-3199</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Federal Shipping</td>
      <td>(503) 555-9931</td>
    </tr>
  </tbody>
</table>
</div>



### Result

Result of the test shows that there is a _**statistically significant**_ difference in performance of shipping companies, hence we reject null hypothesis

# Conclusion

 - Discounts of 5%, 15%, 20% and 25% have approximately the same effect on order quantity
 - Employees from US sold more product with lower discount, though order quantity same as employees from UK and processing time (from order being requested to shipping) approximately the same.
 - There difference in demand of produce, but not significantly enough to reject null hypothesis
 - Discounts were given across categories at the relatively same level

## Further Steps

- Find out why employees from US had much more orders than from UK
- Research further what clients responded better to discount
- Find out optimal level of discount for products according to their price and possible seasonal demand
- Find a way to improve logistics
