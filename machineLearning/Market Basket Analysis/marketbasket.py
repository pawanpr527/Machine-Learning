import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datashader as ds
read_data = pd.read_csv("Online_Retail.csv",encoding='ISO-8859-1')

# Data Cleaning
# print(read_data.isnull().sum())

# x = read_data[(read_data['Description'].isnull()) & (read_data['CustomerID'].isnull())]
# 1454 rows have missing both so we drop them because our dataset is too large so they are neglegible
# print(x.head())

#drop 1454 rows and negative value rows
read_data = read_data.drop(read_data[(read_data['Description'].isnull()) & (read_data['CustomerID'].isnull())].index)
read_data = read_data[read_data['Quantity']>0]
read_data = read_data[read_data['UnitPrice']>0]
read_data['InvoiceDate'] = pd.to_datetime(read_data['InvoiceDate'])
read_data['Date'] = read_data['InvoiceDate'].dt.date
# print(read_data.head(10))
# plt.bar(read_data['InvoiceNo'],read_data['Description'],color='red')
# z = read_data['Description'].value_counts().head(10)

#Count daily Transaction
transaction = read_data.groupby('Date')['InvoiceNo'].nunique()
# print(transaction.max()) #142 transaction in a day 
transaction.plot(kind='line',figsize=(12,5),title='Daily Unique Transaction')

#return per country total transtaction
country = read_data.groupby('Country')['Quantity'].sum().sort_values()
# country.tail(10).plot(kind='bar',title='Country by Quantity')
# plt.show()

#top country by transaction volume
# print(read_data['Country'].value_counts().head(10)) #most of orders from US
# read_data['Country'].value_counts().head(10).plot(kind='bar')

#count total transaction
# print(transaction.sum())

# item = read_data.groupby('Description')['Quantity'].nunique()

#Product level Insights

# product = read_data['Description'].value_counts().head(10)
# print(product)
# colors = [
#     '#1f77b4',  # muted blue
#     '#ff7f0e',  # orange
#     '#2ca02c',  # green
#     '#d62728',  # red
#     '#9467bd',  # purple
#     '#8c564b',  # brown
#     '#e377c2',  # pink
#     '#7f7f7f',  # gray
#     '#bcbd22',  # olive
#     '#17becf'   # cyan
# ]

# product.plot(kind='bar',title="Top 10 Sold Items",figsize=(12,6),legend="Products",color=colors)
# plt.show()

#top 10 sold items by quantity
product = read_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
# print(product)
# product.plot(kind='bar',title='Top 10 item sold by Quantity')
# plt.show()

# product_quantity_distribution = read_data.groupby('Description')['Quantity'].sum().value_counts()
# print(product_quantity_distribution)

read_data['Total_Price'] = read_data['Quantity'] * read_data['UnitPrice']

revenue = read_data.groupby('Description')['Total_Price'].sum().sort_values(ascending=False)
# print(revenue.sum())
# revenue.head(10).plot(kind='bar',title='Top 10 most revenu Product')
# plt.show()

# Date wise revenue

date_wise_revenue = read_data.groupby('Date')['Total_Price'].sum()

# plt.figure(figsize=(12, 6))
# sns.histplot(date_wise_revenue, kde=True, bins=100, color='skyblue', edgecolor='black')
# plt.title("Histogram of Daily Revenue")
# plt.xlabel("Revenue")
# plt.ylabel("Number of Days")
# plt.tight_layout()
# plt.show()
# print(date_wise_revenue.max())
# date_wise_revenue.plot(kind='line',title='day wise Revenue')
# plt.show()
# print(date_wise_revenue)
# print(read_data[read_data['Date']== pd.to_datetime('2011-12-09').date()]['InvoiceNo'].nunique())
# print(read_data['Date'].head())


#  Heatmap of Sales by Hour and Day

read_data['Hour'] = read_data['InvoiceDate'].dt.hour
read_data['DayOfWeek'] = read_data['InvoiceDate'].dt.day_name()

# print(read_data[(read_data['Hour']==10) & (read_data['DayOfWeek']=='Tuesday')]['Total_Price'].sum())
# pivot = read_data.pivot_table(index='Hour', columns='DayOfWeek', values='Total_Price', aggfunc='sum')

# # Reorder days
# days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
# pivot = pivot[days_order]

# plt.figure(figsize=(10,6))
# sns.heatmap(pivot, cmap='YlGnBu')
# plt.title('Revenue Heatmap by Hour and Day of Week')
# plt.show()

# print(read_data.columns)

# print(read_data.head(10))
# print(read_data[read_data['CustomerID']==12347.0].value_counts())

# Customer wise data
read_data = read_data.dropna(subset=['CustomerID'])
# print(read_data[read_data['CustomerID']==12347.0].groupby('InvoiceNo')['Total_Price'].sum().sum())
# print(read_data['Country'].nunique())
customer_df = read_data.groupby('CustomerID').agg({
    'Total_Price' : ['sum','mean'],
    'InvoiceNo' : 'nunique',
    'Quantity' : 'sum',
    'UnitPrice' : 'mean',
    'Country'  : 'first'
}).reset_index()

# Rename columns
customer_df.columns = ['CustomerID', 'TotalSpend', 'AvgSpend', 'NumOrders', 'TotalQty', 'AvgPrice', 'Country']
# print(customer_df.head())

# Encoding or Standardlize feature data
customer_df['Country_Encoded'] = customer_df['Country'].map(customer_df['Country'].value_counts(normalize=True))
# print(customer_df['Country_Encoded'].head(10))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
feature = customer_df.drop(['CustomerID','Country'],axis=1)
scalar = StandardScaler()
scaled = scalar.fit_transform(feature)


''' K-Means, inertia is the total squared distance between each point and its assigned cluster centroid.
It's also called the:
Within-Cluster Sum of Squares (WCSS)
"Compactness" of the clusters 

Inertia vs. k
When k=1, all points belong to 1 cluster → very high inertia

As k increases, points are closer to their centroids → inertia decreases

But beyond some k, the reduction in inertia becomes small → that’s the elbow

Inertia always decreases as you increase the number of clusters k

But increasing k too much:

Makes clusters too small / overfitted

Increases model complexity

Adds no real value after a certain point
'''

#elbow method to find 
# print(scaled.shape)
inertia = []
for k in range(1,11):
    km = KMeans(n_clusters=k,random_state=42)
    km.fit(scaled)
    inertia.append(km.inertia_)
# print(inertia)
# plt.plot(range(1, 11), inertia, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.grid(True)
# plt.show()

#select k =4
kmeans = KMeans(n_clusters=4,random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(scaled)
# print(customer_df['Cluster'].value_counts())
#Summery Cluster
summery = customer_df.groupby('Cluster').agg({
    'TotalSpend' : 'mean',
    'NumOrders': 'mean',
    'TotalQty': 'mean',
    'AvgPrice': 'mean'
})

#Market Basket On per Cluster
from mlxtend.frequent_patterns import apriori,association_rules
df = pd.DataFrame()
def run_apriori(cluster):
    print(f'\n Rules for cluster number {cluster} : \n')

    #customer in cluster
    cluster_cstm = customer_df[customer_df['Cluster']==cluster]['CustomerID']
    cluster_data = read_data[read_data['CustomerID'].isin(cluster_cstm)]
  
    #prepare basket
    basket = cluster_data.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    frquent_item = apriori(basket,min_support=0.02,use_colnames=True)
    rules = association_rules(frquent_item,metric='lift',min_threshold=1.0)
    store_data = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift',ascending=False)
    print(store_data.shape)
    return store_data
    
for cluster in sorted(customer_df['Cluster'].unique()):
   df = pd.concat([df,run_apriori(cluster)],ignore_index=True)
   
df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
df['consequents'] = df['consequents'].apply(lambda x: ', '.join(sorted(x)))
df.to_csv('MBA.csv',index=False,columns=['antecedents', 'consequents'])
