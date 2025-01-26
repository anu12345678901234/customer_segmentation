#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tools
from datetime import date


# In[2]:


#read the data
data = pd.read_excel('C:/Users/anamo/Downloads/Sample Data.xlsx', sheet_name='Sample')


# In[3]:


# print(data.shape)
# print(data['Customer ID'].nunique())


# In[4]:


# print(data['Location'].nunique())


# In[5]:


# data.describe()


# In[6]:


# #Dataset Overview
# dataset_overview = {
#     "Shape": data.shape,
#     "Columns": list(data.columns),
#     "Missing Values": data.isnull().sum().to_dict(),
#     "Duplicate Records": data.duplicated().sum(),
#     "Duplicate Customer IDs": data.duplicated(subset='Customer ID').sum(),
# }

# dataset_overview


# In[7]:


Duplicate = pd.DataFrame(data['Customer ID'].value_counts()).reset_index()
Duplicate.columns = ['Customer ID', 'Records']
(Duplicate)


# In[8]:


data_record_number=data.merge(Duplicate, on='Customer ID')


# In[9]:


data_record_number['Recency'] = (date.today() - data_record_number['Last Purchase Date'].dt.date).dt.days


# In[10]:


data_record_number['rank'] =data.groupby('Customer ID')['Last Purchase Date'].rank( ascending=False)


# In[11]:


data_record_number['Last Purchase Date'] = pd.to_datetime(data_record_number['Last Purchase Date'])

# Rank within each 'Customer ID' by 'Last Purchase Date'
data_record_number['rank'] = data_record_number.groupby('Customer ID')['Last Purchase Date']     .rank(method='dense', ascending=False)

# Filter and reset index for clarity (if needed)
result = data_record_number[data_record_number['Records'] == 2].reset_index()[[
    'Customer ID', 'Age', 'Last Purchase Date', 'Records', 'rank'
]]


# In[12]:


#taking recent purchase
df=data_record_number[data_record_number['rank']==1]


# # Membership Dependency: What factors influence membership?

# In[13]:


# Summary of Membership Tier distribution
membership_distribution = df['Membership Tier'].value_counts().reset_index()

# Display the distribution of membership tiers
membership_distribution


# In[14]:


# Analyze Total Spend across Membership Tiers
spend_by_tier = df.groupby('Membership Tier')['Total Spend'].mean().sort_values(ascending=False)

# Display the average Total Spend per Membership Tier
summary = "\n".join([f"{tier}: {spend:.2f}" for tier, spend in spend_by_tier.items()])
print(f"\nThe average Total Spend across membership tiers is:\n{summary}")

# Optional: Highlight the insights
top_tier = spend_by_tier.idxmax()
print(f"\nInsight: {top_tier} members tend to spend the most on average.")
spend_by_tier=pd.DataFrame(spend_by_tier).reset_index()


# In[15]:


import matplotlib.pyplot as plt

# Plot Total Spend by Membership Tier
plt.figure(figsize=(4, 2))
spend_by_tier.plot(kind='bar', color='darkblue', edgecolor='black')
plt.title('Average Total Spend by Membership Tier')
plt.xlabel('Membership Tier')
plt.ylabel('Average Total Spend')
plt.xticks(rotation=0)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[16]:


# Analyze Purchase Frequency across Membership Tiers
frequency_distribution = df.groupby(['Membership Tier', 'Purchase Frequency']).size().unstack()

# Display the distribution of Purchase Frequency per Membership Tier
print(frequency_distribution)
print('\n\nInsight:')
print('Gold members have a relatively higher proportion of "High" purchase frequency compared to other tiers.')

# Plot Purchase Frequency distribution by Membership Tier as a clustered column chart
fig, ax = plt.subplots(figsize=(4, 6))

colors = ['darkblue', 'steelblue', 'lightgrey']
frequency_distribution.plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.8)

# Add labels at the end of each bar
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
                    ha='center', va='bottom', rotation=90, fontsize=9)

# Customize plot aesthetics
ax.set_title('Purchase Frequency Distribution by Membership Tier')
ax.set_xlabel('Membership Tier')
ax.set_ylabel('Count of Customers')
ax.legend(title='Purchase Frequency', loc='upper right')
#ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.show()


# In[17]:


frequency_distribution=pd.DataFrame(frequency_distribution).reset_index()


# In[18]:


# Analyze Promo Response Rate across Membership Tiers
promo_response_by_tier = df.groupby('Membership Tier')['Promo Response Rate'].mean().sort_values(ascending=False)

# Display the average Promo Response Rate per Membership Tier
print(promo_response_by_tier)
print('\n\nInsight:')
print(' Silver members respond to promotions slightly more frequently than Gold and Bronze members.')


# In[19]:


promo_response_by_tier=pd.DataFrame(promo_response_by_tier).reset_index()


# In[20]:


# Analyze Feedback Rating across Membership Tiers
feedback_by_tier = df.groupby('Membership Tier')['Feedback Rating'].mean().sort_values(ascending=False)

# Display the average Feedback Rating per Membership Tier
print(feedback_by_tier)

print('\n\nInsight:')
print('-Silver and Gold members provide slightly higher feedback ratings than Bronze members, though the differences are marginal.','\n-Silver members exhibit the highest feedback ratings, indicating relatively higher satisfaction or engagement compared to Bronze and Gold members.')


# In[21]:


feedback_by_tier=pd.DataFrame(feedback_by_tier).reset_index()


# Observations:
# 
# Total Spend and Purchase Frequency show notable increases with higher membership tiers.
# 
# Promo Response Rate and Feedback Ratings are slightly better for Silver members compared to Gold and Bronze.

# # Support Analysis
# 

# In[22]:


#Identify the age group needing more support based on the number of support interactions.

# Define age groups based on the given schema
def categorize_age(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 55:
        return "46-55"
    elif age <= 65:
        return "56-65"
    else:
        return "66+"

# Add an Age Group column to the dataset
df['Age Group'] = df['Age'].apply(categorize_age)

# Analyze support interactions by age group
support_by_age_group = df.groupby('Age Group')['Support Interactions'].sum().sort_values(ascending=False)

# Display support interactions by age group
print(support_by_age_group)

print('\n\nInsight:')
print('The 46-55 age group requires the most support.')
support_by_age_group =pd.DataFrame(support_by_age_group ).reset_index()


# In[23]:


#Analyze spending distribution for the group whose issues are resolved.

# Filter data for the 46-55 age group and resolved issues
resolved_issues_46_55 = df[(df['Age Group'] == '46-55') & (df['Issues Resolved'] == 'Yes')]

# Analyze Total Spend distribution
spend_distribution = resolved_issues_46_55['Total Spend'].describe()

# Display spending distribution for resolved issues in the 46-55 age group
print(spend_distribution)
spend_distribution=pd.DataFrame(spend_distribution).reset_index

print('\n\nInsight:')
print('The 46-55 age group requires the most support.')


# In[24]:


Current_date = pd.to_datetime('2025-01-18')
df['Last Purchase Date'] = pd.to_datetime(df['Last Purchase Date'])
df['Recency'] = (Current_date - df['Last Purchase Date']).dt.days

# Define Customer Status based on Recency
def customer_status(recency):
    if recency <= 90:
        return "Active"
    elif recency <= 180:
        return "Less Active (91-180)"
    elif recency <= 365:
        return "Very Less Active (181-365)"
    else:
        return "Lapsed"


# In[25]:


df['Customer Status'] = df['Recency'].apply(customer_status)
# Analyze activity status for resolved issues in the 46-55 age group
df['Customer Status']


# In[26]:


status_distribution = df['Customer Status'].value_counts()
status_distribution =pd.DataFrame(status_distribution ).reset_index()
# Display activity status distribution
status_distribution


# In[27]:


status_distribution_across = pd.DataFrame(df.groupby(df['Customer Status'])['Total Spend'].sum()).reset_index()
status_distribution_across['PercentSpend']=status_distribution_across['Total Spend']/sum(status_distribution_across['Total Spend'])

# Display activity status distribution
CustomerStatusbyTotalSpend=pd.DataFrame(status_distribution_across.sort_values(by='Total Spend',ascending=False))
status_distribution_across.sort_values(by='Total Spend',ascending=False)


# In[28]:



status_distribution_across = (
    df.groupby(['Customer Status', 'Age Group', 'Issues Resolved'])['Total Spend']
    .sum()
    .reset_index()
)


status_distribution_across['PercentSpend']=status_distribution_across['Total Spend']/sum(status_distribution_across['Total Spend'])
status_distribution=pd.DataFrame(status_distribution_across.sort_values(by=['Issues Resolved','Age Group','Total Spend'],ascending=False))
# Display activity status distribution
status_distribution_across.sort_values(by=['Issues Resolved','Age Group','Total Spend'],ascending=False)


# # The table shows the total spending and percentage spending distribution across Customer Status, Age Group, and Issues Resolved. Key insights from this data include:
# 
# 66+ Age Group:
# 
# Customers who are Very Less Active (181-365) contribute the highest spend among resolved issues in this group, making up 2.98% of total spending.
# Even among Less Active (91-180) and Active (≤90 days) customers, spending remains substantial.
# 
# 56-65 Age Group:
# 
# Very Less Active (181-365) customers contribute the highest total spend, forming 5.58% of the total spending, emphasizing that resolving issues for this group has a significant impact on spending.
# 
# Overall Spending:
# 
# Despite resolving issues, customers classified as Very Less Active contribute a disproportionate amount of spending compared to Active customers.

# In[29]:


# Analyze feedback ratings for resolved issues in the 46-55 age group
feedback_distribution = resolved_issues_46_55['Feedback Rating'].describe()

# Display feedback rating distribution
feedback_distribution


# In[30]:


feedback_distribution46_55=pd.DataFrame(feedback_distribution).reset_index()


# In[31]:


# Analyze feedback ratings for resolved issues in the 46-55 age group
feedback_distribution =df['Feedback Rating'].describe()

# Display feedback rating distribution
feedback_distribution


# In[32]:


feedback_distribution=pd.DataFrame(feedback_distribution).reset_index()


# In[33]:


print('The proportion of customers with unresolved issues across Customer Status categories:')
# Analyze unresolved issues by Customer Status
unresolved_issues = df[df['Issues Resolved'] == 'No']

# Calculate the distribution of Customer Status for unresolved issues
unresolved_status_distribution = unresolved_issues['Customer Status'].value_counts(normalize=True) * 100

# Display the distribution of unresolved issues across customer statuses
print(unresolved_status_distribution)

print('\n\nInsight:')
print('This indicates that unresolved issues are disproportionately associated with "Very Less Active" customers, suggesting a potential link between unresolved issues and inactivity.')


# In[34]:


unresolved_issues=pd.DataFrame(unresolved_issues).reset_index()


# In[35]:



print('Total spending and activity levels (Active, Less Active, Very Less Active, Lapsed) for customers with "No" issue resolution: \n')
# Analyze total spending for unresolved issues across Customer Status
unresolved_spend_distribution = unresolved_issues.groupby('Customer Status')['Total Spend'].sum()

# Display spending distribution for unresolved issues
print(unresolved_spend_distribution)

print('\n\nInsight:')
print('Very Less Active customers contribute the most spending, totaling 744,149.','\nSpending significantly drops for customers categorized as Lapsed, indicating a reduced engagement post-inactivity.')


# In[36]:


unresolved_spend_distribution=pd.DataFrame(unresolved_spend_distribution).reset_index()


# In[ ]:





# In[37]:


print('Feedback ratings for customers with unresolved issues to understand their satisfaction levels.:\n')
# Analyze feedback ratings for unresolved issues across Customer Status
unresolved_feedback_distribution = unresolved_issues.groupby('Customer Status')['Feedback Rating'].mean()

# Display feedback ratings for unresolved issues
print(unresolved_feedback_distribution)

print('\n\nInsight:')
print('Feedback ratings are generally moderate (~3.0) for all customer statuses except Lapsed, where the rating drops significantly to 2.46, indicating dissatisfaction.')


# In[38]:


unresolved_feedback_distribution=pd.DataFrame(unresolved_feedback_distribution).reset_index()


# "Very Less Active (181-365)" customers account for 49.50% of unresolved issues, making them the most impacted group.
# 
# These customers contribute the highest total spending (744,149), indicating significant potential for re-engagement if their issues are resolved.
# 
# Lapsed customers have the lowest feedback ratings (2.46), highlighting the adverse effects of unresolved issues on both satisfaction and retention.

# In[39]:


# Analyze Purchase Frequency distribution across Age Groups in the filtered dataset
frequency_by_age_group = df.groupby(['Age Group', 'Purchase Frequency']).size().unstack(fill_value=0)

# Display the distribution of Purchase Frequency by Age Group
print(frequency_by_age_group)
print("\n\nInsights:")
print("The 46-55 age group consistently has the largest user base across all frequency categories (High, Medium, Low), highlighting this group as a key target for retention and engagement strategies.")


# In[40]:


frequency_by_age_group=pd.DataFrame(frequency_by_age_group).reset_index()


# In[41]:


# Analyze average Time on Site by Age Group in the filtered dataset
time_on_site_by_age_group = df.groupby('Age Group')['Time on Site (min)'].mean()

# Analyze Promo Response Rate by Age Group in the filtered dataset
promo_response_by_age_group = df.groupby('Age Group')['Promo Response Rate'].mean()

# Combine both metrics for easy comparison
time_promo_summary = pd.DataFrame({
    'Average Time on Site (min)': time_on_site_by_age_group,
    'Average Promo Response Rate': promo_response_by_age_group
})

# Display the results
print(time_promo_summary)

print("\n\nInsights:")
print("Younger age groups (18-25 and 26-35) spend slightly less time on the site compared to older groups like 46-55.","\nPromo response rates are relatively consistent across age groups, with the 26-35 age group having the highest rate.")


# In[42]:


time_on_site_by_age_group=pd.DataFrame(time_on_site_by_age_group).reset_index()


# In[43]:


print("Which age group calls support repetitively?\n")
support_interactions_by_age_group =df .groupby('Age Group')['Support Interactions'].sum()
# Total support interactions for all age groups
total_support_interactions = support_interactions_by_age_group.sum()

# Calculate percentage contribution of support interactions by age group
support_interactions_percentage = (support_interactions_by_age_group / total_support_interactions) * 100

# Combine absolute numbers and percentages for clarity
support_interactions_summary = pd.DataFrame({
    'Total Interactions': support_interactions_by_age_group,
    'Percentage (%)': support_interactions_percentage
})

# Plot the results
plt.figure(figsize=(4, 3))
support_interactions_by_age_group.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Total Support Interactions by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Support Interactions')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Display the data for clarity
print(support_interactions_summary)

print("\n\nInsights:")
print("The 46-55 age group accounts for the highest number of support interactions:\n857 interactions, which is 21.48% of total interactions.\nThe 36-45 age group follows with 19.18%, while 26-35 accounts for 17.17%.\nThe 18-25 age group has the fewest interactions relative to others, at 14.92%.")


# In[44]:


support_interactions_summary=pd.DataFrame(support_interactions_summary).reset_index()


# In[45]:


# Calculate percentage contribution for Purchase Frequency in the 46-55 age group
purchase_frequency_46_55 = df[df['Age Group'] == '46-55']['Purchase Frequency'].value_counts()
purchase_frequency_percentage_46_55 = (purchase_frequency_46_55 / purchase_frequency_46_55.sum()) * 100

# Combine absolute numbers and percentages for clarity
purchase_frequency_summary_46_55 = pd.DataFrame({
    'Total Customers': purchase_frequency_46_55,
    'Percentage (%)': purchase_frequency_percentage_46_55
})

# Plot the results
plt.figure(figsize=(4, 3))
purchase_frequency_46_55.plot(kind='bar', color=['darkblue', 'steelblue', 'lightgrey'], edgecolor='black')
plt.title('Purchase Frequency Distribution for 46-55 Age Group')
plt.xlabel('Purchase Frequency')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Display the results
print(purchase_frequency_summary_46_55)

print("\n\nInsights:")
print("Among the 46-55 age group (frequent support callers):\n 41.06% are Low Frequency buyers (170 customers).\n 38.89% are Medium Frequency buyers (161 customers).\n 20.05% are High Frequency buyers (83 customers).")


# In[46]:


purchase_frequency_summary_46_55=pd.DataFrame(purchase_frequency_summary_46_55).reset_index()


# In[47]:



print("Examine whether these frequent support callers are active or inactive based on Customer StatuS")
customer_status_46_55 =df[df['Age Group'] == '46-55']['Customer Status'].value_counts()


# Calculate percentage contribution for Customer Status in the 46-55 age group
customer_status_percentage_46_55 = (customer_status_46_55 / customer_status_46_55.sum()) * 100

# Combine absolute numbers and percentages for clarity
customer_status_summary_46_55 = pd.DataFrame({
    'Total Customers': customer_status_46_55,
    'Percentage (%)': customer_status_percentage_46_55
})

# Plot the results
plt.figure(figsize=(4, 2))
customer_status_46_55.plot(kind='bar', color=['darkblue', 'steelblue', 'lightgrey', 'grey'], edgecolor='black')
plt.title('Customer Status Distribution for 46-55 Age Group')
plt.xlabel('Customer Status')
plt.ylabel('Number of Customers')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Display the results
print(customer_status_summary_46_55)

print("\n\nInsights:")
print("Among the 46-55 age group:\n-48.79% are classified as Very Less Active (181-365 days) (202 customers).\n-23.91% are Active (≤90 days) and Less Active (91-180 days), each with 99 customers.\n-Only 3.38% are Lapsed (>365 days) (14 customers).")


# In[48]:


customer_status_summary_46_55=pd.DataFrame(customer_status_summary_46_55).reset_index()


# In[49]:


# Filter data for Home & Kitchen and Beauty categories
home_spending = df[df['Preferred Category'] == 'Home & Kitchen']['Total Spend']
beauty_spending = df[df['Preferred Category'] == 'Beauty']['Total Spend']

# Calculate average spending for each category
average_spending_home = home_spending.mean()
average_spending_beauty = beauty_spending.mean()

# Combine results for comparison
average_spending_comparison = pd.DataFrame({
    'Category': ['Home & Kitchen', 'Beauty'],
    'Average Spending': [average_spending_home, average_spending_beauty]
})

# Calculate percentage difference between the categories
total_average_spending = average_spending_home + average_spending_beauty
home_percentage = (average_spending_home / total_average_spending) * 100
beauty_percentage = (average_spending_beauty / total_average_spending) * 100

# Update the comparison table with percentages
average_spending_comparison['Percentage (%)'] = [home_percentage, beauty_percentage]

# Display the results
print(average_spending_comparison)

print("\n\nInsights:")
print("1. Average Spending:\n   - Home & Kitchen: $2,435.71 (49.21% of total average spending).\n   - Beauty: $2,513.99 (50.79% of total average spending).\n2. Comparison:\n   - The Beauty category accounts for a slightly higher proportion of spending compared to the **Home & Kitchen** category.")


# In[50]:


average_spending_comparison=pd.DataFrame(average_spending_comparison).reset_index()


# In[51]:


# Analyze category preferences by gender
category_preferences_by_gender = df.groupby(['Gender', 'Preferred Category']).size().unstack(fill_value=0)

# Calculate percentages by gender for each category
category_preferences_percentage = category_preferences_by_gender.div(category_preferences_by_gender.sum(axis=1), axis=0) * 100

# Combine absolute numbers and percentages for a comprehensive view
category_preferences_summary = category_preferences_by_gender.copy()
for gender in category_preferences_by_gender.index:
    category_preferences_summary.loc[gender] = [
        f"{int(category_preferences_by_gender.loc[gender, cat])} ({category_preferences_percentage.loc[gender, cat]:.2f}%)"
        for cat in category_preferences_by_gender.columns
    ]

# Display the results
print(pd.DataFrame(category_preferences_summary).transpose())

print("\n\nInsights:")
print("Most preferred category for female is: Electronics (21.90%, 228 instances) while for men is Sports & Outdoors (22.04%, 210 instances)")


# In[52]:


category_preferences_summary=pd.DataFrame(category_preferences_summary).transpose().reset_index()


# In[53]:


print("Under each gender, which age group shows higher preference for certain categories?")
# Analyze preferences by Gender, Age Group, and Preferred Category
preferences_by_gender_age_category = df.groupby(
    ['Gender', 'Age Group', 'Preferred Category']
).size().unstack(fill_value=0)

# Calculate percentages within each Gender and Age Group
preferences_percentage_by_gender_age = preferences_by_gender_age_category.div(
    preferences_by_gender_age_category.sum(axis=1), axis=0
) * 100

# Combine absolute numbers and percentages for clarity
preferences_summary = preferences_by_gender_age_category.copy()
for idx in preferences_by_gender_age_category.index:
    preferences_summary.loc[idx] = [
        f"{int(preferences_by_gender_age_category.loc[idx, cat])} ({preferences_percentage_by_gender_age.loc[idx, cat]:.2f}%)"
        for cat in preferences_by_gender_age_category.columns
    ]

# Display the detailed results
print(preferences_summary.transpose())


# In[54]:


preferences_summary=pd.DataFrame(preferences_summary.transpose()).reset_index()


# In[55]:


# Recalculate support interactions and resolution analysis using the latest transactions data

# Total support interactions by category
support_by_category_latest = df.groupby('Preferred Category')['Support Interactions'].sum()

# Issues resolved analysis by category
issues_resolved_by_category_latest = df.groupby(
    ['Preferred Category', 'Issues Resolved']
).size().unstack(fill_value=0)

# Calculate percentage of resolved issues for each category
resolved_percentage_latest = (
    (issues_resolved_by_category_latest['Yes'] / issues_resolved_by_category_latest.sum(axis=1)) * 100
)

# Calculate percentage of total support interactions for each category
total_support_interactions_latest = support_by_category_latest.sum()
support_percentage_latest = (support_by_category_latest / total_support_interactions_latest) * 100

# Combine results into a summary table
support_analysis_summary_latest = pd.DataFrame({
    'Total Support Interactions': support_by_category_latest,
    'Percentage of Total (%)': support_percentage_latest.round(2),
    'Resolved Issues (%)': resolved_percentage_latest.round(2)
}).fillna(0)

# Display the updated summary
(support_analysis_summary_latest.transpose())


# In[56]:


support_analysis_summary_latest=pd.DataFrame(support_analysis_summary_latest.transpose()).reset_index()


# In[57]:


# Analyze categories with high promo responses
# Define a threshold for "high promo response rate" (e.g., above 75th percentile)
promo_response_threshold = df['Promo Response Rate'].quantile(0.75)
high_promo_data = df[df['Promo Response Rate'] > promo_response_threshold]

# Calculate averages for metrics of interest (Pages Visited, Time on Site)
avg_metrics_high_promo = high_promo_data.groupby('Preferred Category')[ ['Pages Visited', 'Time on Site (min)']].mean()

# Calculate percentage of "early offers" (1-10 days in a month)
# Correcting the analysis to focus on purchase timing instead of offer timing
# Calculate purchase period based on the day of the last purchase
high_promo_data['Purchase Period'] = pd.to_datetime(high_promo_data['Last Purchase Date']).dt.day.apply(
    lambda x: 'Early' if x <= 10 else ('Mid' if x <= 20 else 'End')
)

# Analyze the distribution of purchases across periods for high promo response customers
purchase_period_distribution = high_promo_data.groupby('Purchase Period').size() / high_promo_data.shape[0] * 100

# Display average metrics for high promo response customers by category
avg_metrics_high_promo = high_promo_data.groupby('Preferred Category')[
    ['Pages Visited', 'Time on Site (min)']
].mean()



# Visualization of purchase period distribution
plt.figure(figsize=(4, 2))
purchase_period_distribution.plot(kind='bar', color=['steelblue', 'lightblue', 'grey'], edgecolor='black')
plt.title('Purchase Period Distribution for High Promo Response')
plt.xlabel('Purchase Period')
plt.ylabel('Percentage of Customers')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


(avg_metrics_high_promo)


# In[58]:


avg_metrics_high_promo=pd.DataFrame(avg_metrics_high_promo).reset_index()


# In[59]:


# Analyze the relationship between time spent on the site and total spending
# Correlation between Time on Site and Total Spend
correlation_time_spend = df['Time on Site (min)'].corr(df['Total Spend'])

# Group data by quantiles of Time on Site to analyze average spending
time_spent_quantiles = pd.qcut(df['Time on Site (min)'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
avg_spending_by_time = df.groupby(time_spent_quantiles)['Total Spend'].mean()

# Calculate the percentage contribution of each group to total spend
total_spending = df['Total Spend'].sum()
spending_percentage_by_time = (df.groupby(time_spent_quantiles)['Total Spend'].sum() / total_spending) * 100

# Combine results into a summary table
time_spending_summary = pd.DataFrame({
    'Average Spending': avg_spending_by_time,
    'Percentage of Total Spending (%)': spending_percentage_by_time
}).round(2)



# Visualization of spending by time quantiles
plt.figure(figsize=(4, 2))
avg_spending_by_time.plot(kind='bar', color=['steelblue', 'lightblue', 'grey', 'darkblue'], edgecolor='black')
plt.title('Average Spending by Time Spent on Site')
plt.xlabel('Time Spent on Site (Quantiles)')
plt.ylabel('Average Spending ($)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

(time_spending_summary)


# In[60]:


time_spending_summary=pd.DataFrame(time_spending_summary).reset_index()


# In[61]:


# Calculate correlations
correlation_promo_pages = df['Promo Response Rate'].corr(df['Pages Visited'])
correlation_promo_time = df['Promo Response Rate'].corr(df['Time on Site (min)'])

# Prepare a summary of correlations
correlation_summary = pd.DataFrame({
    'Metric': ['Pages Visited', 'Time on Site (min)'],
    'Correlation with Promo Response Rate': [correlation_promo_pages, correlation_promo_time]
}).round(2)

# Display the summary
(correlation_summary)


# In[62]:


correlation_summary=pd.DataFrame(correlation_summary).reset_index()


# In[63]:


# Group data by Purchase Frequency and calculate averages for Pages Visited and Time on Site
purchase_frequency_metrics = df.groupby('Purchase Frequency')[
    ['Pages Visited', 'Time on Site (min)']
].mean().round(2)

# Add percentages of total customers in each frequency category
frequency_distribution = df['Purchase Frequency'].value_counts(normalize=True) * 100
purchase_frequency_metrics['Percentage of Customers (%)'] = frequency_distribution

# Display the results
(purchase_frequency_metrics.transpose())


# In[64]:


purchase_frequency_metrics=pd.DataFrame(purchase_frequency_metrics.transpose()).reset_index()


# In[65]:


# Group by Age Group and calculate average Total Spend and Support Interactions
support_spending_analysis = df.groupby('Age Group')[
    ['Total Spend', 'Support Interactions']
].mean().round(2)

# Calculate correlation between Support Interactions and Total Spend for the entire dataset
correlation_support_spend = df['Support Interactions'].corr(df['Total Spend'])

# Add correlation to the summary
support_spending_analysis['Correlation (Support vs Spend)'] = correlation_support_spend

# Display the final output
support_spending_analysis


# In[66]:


support_spending_analysis=pd.DataFrame(support_spending_analysis).reset_index()


# In[67]:


# Step 2: Filter "High Frequency" customers and calculate the average spend
high_frequency_customers = df[df["Purchase Frequency"] == "High"]
average_spend_high_frequency = high_frequency_customers["Total Spend"].mean()

# Display the calculated average spend
print('The average spend of "High Frequency" customers is:')
average_spend_high_frequency


# In[68]:



high_frequency_customers = df
average_spend_per_frequency = high_frequency_customers.groupby("Purchase Frequency")["Total Spend"].mean()

# Display the calculated average spend
print('The average spend of "High Frequency" customers is:')
average_spend_per_frequency=pd.DataFrame(average_spend_per_frequency).reset_index()


# In[69]:


# Step 3: Analyze the most preferred category among high-frequency customers
preferred_category_counts = high_frequency_customers["Preferred Category"].value_counts()

# Identify the most preferred category
most_preferred_category = preferred_category_counts.idxmax()
most_preferred_count = preferred_category_counts.max()

# Plot the distribution of preferred categories
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 2))
preferred_category_counts.plot(kind="bar", color=plt.cm.Blues(range(len(preferred_category_counts))))
plt.title("Preferred Categories Among High-Frequency Customers", fontsize=12)
plt.xlabel("Preferred Category", fontsize=8)
plt.ylabel("Number of Customers", fontsize=8)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

most_preferred_category, most_preferred_count


# In[70]:


preferred_category_counts=pd.DataFrame(preferred_category_counts).reset_index()


# In[71]:


#During which month is the promo response rate highest?
# Step 1: Extract the month from the "Last Purchase Date"
df["Purchase Month"] = pd.to_datetime(df["Last Purchase Date"]).dt.month

# Step 2: Calculate the average promo response rate by month
monthly_promo_response = df.groupby("Purchase Month")["Promo Response Rate"].mean()

# Step 3: Plot the monthly promo response rates
plt.figure(figsize=(4, 3))
monthly_promo_response.plot(kind="bar", color=plt.cm.Blues(range(len(monthly_promo_response))))
plt.title("Average Promo Response Rate by Month", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Promo Response Rate", fontsize=12)
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Identify the month with the highest promo response rate
highest_promo_month = monthly_promo_response.idxmax()
highest_promo_rate = monthly_promo_response.max()

print(highest_promo_month, highest_promo_rate)

monthly_promo_response=pd.DataFrame(monthly_promo_response).reset_index()
monthly_promo_response.reset_index()


# Month with the Highest Promo Response Rate:
# 
# The highest average promo response rate occurs in April, with a rate of 57.70%.
# 
# The bar chart above illustrates the average promo response rates across all month

# In[72]:


#Which categories are purchased more in specific months?
# Step 4: Analyze categories purchased by month
monthly_category_counts = df.groupby(["Purchase Month", "Preferred Category"]).size().unstack(fill_value=0)

# Plot the category distribution for specific months
plt.figure(figsize=(4, 3))
monthly_category_counts.plot(kind="bar", stacked=True, cmap="Blues", figsize=(8, 4))
plt.title("Category Purchases by Month", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Purchases", fontsize=12)
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Identify the most purchased category in each month
most_purchased_categories = monthly_category_counts.idxmax(axis=1)
print(most_purchased_categories.head())
monthly_category_counts


# In[73]:


monthly_category_counts=pd.DataFrame(monthly_category_counts).reset_index()


# In[74]:


# Step 5: Analyze spending at different periods of the month
# Create a "Period of the Month" column based on the day of the month
filtered_data=df
filtered_data["Day of Month"] = pd.to_datetime(df["Last Purchase Date"]).dt.day
filtered_data["Period of Month"] = pd.cut(
    filtered_data["Day of Month"],
    bins=[0, 10, 20, 31],
    labels=["Early", "Mid", "End"],
    right=True
)

# Calculate average spend by period of the month
average_spend_period = filtered_data.groupby("Period of Month")["Total Spend"].mean()

# Plot average spend for different periods
plt.figure(figsize=(4, 2))
average_spend_period.plot(kind="bar", color=plt.cm.Blues(range(len(average_spend_period))))
plt.title("Average Spend by Period of the Month", fontsize=14)
plt.xlabel("Period of the Month", fontsize=12)
plt.ylabel("Average Spend", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Identify the period with the highest spend
highest_spend_period = average_spend_period.idxmax()
highest_spend_value = average_spend_period.max()
print('Do customers spend more at the start or end of the month?')
print(highest_spend_period, highest_spend_value)

average_spend_period.reset_index()


# In[75]:


# Step 6: Analyze categories purchased during different periods of the month
period_category_counts = filtered_data.groupby(["Period of Month", "Preferred Category"]).size().unstack(fill_value=0)

# Plot category distribution for different periods of the month
plt.figure(figsize=(8, 4))
period_category_counts.plot(kind="bar", stacked=True, cmap="Blues", figsize=(8, 4))
plt.title("Category Purchases by Period of the Month", fontsize=14)
plt.xlabel("Period of the Month", fontsize=12)
plt.ylabel("Number of Purchases", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Identify the most purchased category for each period
most_purchased_period_categories = period_category_counts.idxmax(axis=1)
print(most_purchased_period_categories.head())
period_category_counts.reset_index()


# Categories Purchased During Different Periods of the Month
# Early (1st–10th):
# Most purchased category: Sports & Outdoors with 149 purchases.
# Accounts for 23.4% of purchases in the early period.
# 
# Mid (11th–20th):
# Most purchased category: Beauty with 141 purchases.
# Accounts for 21.8% of purchases in the mid period.
# 
# End (21st–end of the month):
# Most purchased category: Sports & Outdoors again, with 151 purchases.
# Accounts for 23.7% of purchases in the end period.

# In[76]:


# Step 9: Analyze the promo response rate within periods of the month


updated_promo_response_period = filtered_data.groupby("Period of Month")["Promo Response Rate"].mean()

# Plot the updated promo response rates
plt.figure(figsize=(4, 3))
updated_promo_response_period.plot(kind="bar", color=plt.cm.Blues(range(len(updated_promo_response_period))))
plt.title("Promo Response Rate by Period of the Month", fontsize=14)
plt.xlabel("Period of the Month", fontsize=12)
plt.ylabel("Average Promo Response Rate", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Identify the period with the highest promo response rate
highest_updated_promo_period = updated_promo_response_period.idxmax()
highest_updated_promo_rate = updated_promo_response_period.max()

print(highest_updated_promo_period, highest_updated_promo_rate)
updated_promo_response_period.reset_index()


# In[77]:


# Step 10: Recalculate customer activity by period of the month
customer_activity_period = filtered_data.groupby("Period of Month")["Customer ID"].count()

# Plot customer activity levels for different periods of the month
plt.figure(figsize=(4, 3))
customer_activity_period.plot(kind="bar", color=plt.cm.Blues(range(len(customer_activity_period))))
plt.title("Customer Activity by Period of the Month", fontsize=14)
plt.xlabel("Period of the Month", fontsize=12)
plt.ylabel("Number of Transactions", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Identify periods of high and low activity
highest_activity_period = customer_activity_period.idxmax()
lowest_activity_period = customer_activity_period.idxmin()
highest_activity_count = customer_activity_period.max()
lowest_activity_count = customer_activity_period.min()

print(highest_activity_period, highest_activity_count, lowest_activity_period, lowest_activity_count)
customer_activity_period.reset_index()


# In[78]:


# Analyzing customer activity and category preferences in alignment with seasonal behaviors

# Step 1: Calculate monthly customer activity
monthly_customer_activity = filtered_data.groupby(filtered_data["Purchase Month"])["Customer ID"].count()

# Step 2: Calculate category purchases by month
monthly_category_purchases = filtered_data.groupby(["Purchase Month", "Preferred Category"]).size().unstack(fill_value=0)

# Step 3: Identify peak months for activity and category preferences
peak_month = monthly_customer_activity.idxmax()
lowest_month = monthly_customer_activity.idxmin()

# Step 4: Aggregate data for seasonal analysis
monthly_data_summary = monthly_category_purchases.sum(axis=1).reset_index()
monthly_data_summary.columns = ["Month", "Total Transactions"]
monthly_data_summary["Percentage of Total"] = (
    monthly_data_summary["Total Transactions"] / monthly_data_summary["Total Transactions"].sum() * 100
)



# Plot customer activity by month
plt.figure(figsize=(4, 3))
monthly_customer_activity.plot(kind="line", marker="o", color="blue", linewidth=2)
plt.title("Monthly Customer Activity", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Transactions", fontsize=12)
plt.xticks(ticks=range(1, 13), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Output peak and low activity months
print(peak_month, monthly_customer_activity[peak_month], lowest_month, monthly_customer_activity[lowest_month])
# Display monthly activity and category data
monthly_data_summary


# In[79]:


monthly_data_summary=pd.DataFrame(monthly_data_summary).reset_index()


# # RFM ANALYSIS

# In[80]:


# Step 1: Calculate RFM metrics

# Recency: Days since last purchase (using the current date: 2025-01-18)
filtered_data["Recency"] = (pd.Timestamp("2025-01-18") - pd.to_datetime(filtered_data["Last Purchase Date"])).dt.days

# Frequency: Categorical values (already available in the "Purchase Frequency" column)
# Map categorical frequency to numerical scale (Low = 1, Medium = 2, High = 3 for analysis purposes)
frequency_mapping = {"Low": 1, "Medium": 2, "High": 3}
filtered_data["Frequency"] = filtered_data["Purchase Frequency"].map(frequency_mapping)

# Monetary Value: Total Spend
filtered_data["Monetary"] = filtered_data["Total Spend"]

# Step 2: Create RFM segmentation by scoring Recency, Frequency, and Monetary
# Recency: Lower value is better (closer to today)
filtered_data["Recency_Score"] = pd.qcut(filtered_data["Recency"], 4, labels=[4, 3, 2, 1])

# Frequency: Higher value is better (mapped from categories)
filtered_data["Frequency_Score"] = filtered_data["Frequency"]

# Monetary: Higher value is better
filtered_data["Monetary_Score"] = pd.qcut(filtered_data["Monetary"], 4, labels=[1, 2, 3, 4])

# Step 3: Calculate RFM score
filtered_data["RFM_Score"] = (
    filtered_data["Recency_Score"].astype(int) +
   filtered_data["Frequency_Score"].astype(int) +
    filtered_data["Monetary_Score"].astype(int)
)

# Step 4: Segment customers based on RFM score
def segment_customer(rfm_score):
    if rfm_score >= 9:
        return "Champions"
    elif rfm_score >= 7:
        return "Loyal Customers"
    elif rfm_score >= 5:
        return "Potential Loyalists"
    elif rfm_score >= 4:
        return "Needs Attention"
    else:
        return "At Risk"

filtered_data["Segment"] = filtered_data["RFM_Score"].apply(segment_customer)

# Display summarized RFM segments
rfm_segments = filtered_data["Segment"].value_counts().reset_index()
rfm_segments.columns = ["Segment", "Customer Count"]
rfm_segments['Distribution']=rfm_segments["Customer Count"]/1994
(rfm_segments)


# In[81]:


bins = [0, 0.25, 0.50, 0.75, 1.00]
labels = ['Low', 'Medium', 'High', 'Very High']
filtered_data['Promo Response Category'] = pd.cut(filtered_data['Promo Response Rate'], bins=bins, labels=labels)
loyal_customers_segment = filtered_data[ filtered_data['Segment'] == 'Loyal Customers']


# In[82]:


# Filter data for the "Loyal Customers" segment
loyal_customers_segment = filtered_data[ filtered_data['Segment'] == 'Loyal Customers']

# Summary statistics for key metrics related to engagement and retention
loyal_customers_summary = loyal_customers_segment[
    ['Total Spend', 'Purchase Frequency', 'Pages Visited', 'Time on Site (min)', 'Promo Response Category']
].describe()

# Count of promo response categories for the Loyal Customers segment
promo_response_counts_loyal = loyal_customers_segment['Promo Response Category'].value_counts()

# Analyze purchase frequency distribution for Loyal Customers
purchase_frequency_distribution = loyal_customers_segment['Purchase Frequency'].value_counts()

loyal_customers_summary=pd.DataFrame(loyal_customers_summary).reset_index()
promo_response_counts_loyal=pd.DataFrame(promo_response_counts_loyal).reset_index()
purchase_frequency_distribution=pd.DataFrame(purchase_frequency_distribution).reset_index()

loyal_customers_summary, promo_response_counts_loyal, purchase_frequency_distribution


# In[83]:


# Filter data for the "Needs Attention" segment
needs_attention_segment = filtered_data[filtered_data['Segment'] == 'Needs Attention']

# Summary statistics for key metrics related to engagement and retention
needs_attention_summary = needs_attention_segment[
    ['Total Spend', 'Purchase Frequency', 'Pages Visited', 'Time on Site (min)', 'Promo Response Category']
].describe()

# Count of promo response categories for the Needs Attention segment
needs_attention_promo_response_counts = needs_attention_segment['Promo Response Category'].value_counts()

# Distribution of purchase frequency for Needs Attention
needs_attention_purchase_frequency_distribution = needs_attention_segment['Purchase Frequency'].value_counts()

need_attention_summary=pd.DataFrame(needs_attention_summary).reset_index()
pr_counts_need_attention=pd.DataFrame(needs_attention_promo_response_counts).reset_index()
fd_need_attention=pd.DataFrame(needs_attention_purchase_frequency_distribution).reset_index()

# Display the results
needs_attention_summary, needs_attention_promo_response_counts, needs_attention_purchase_frequency_distribution


# In[84]:


# Filter data for the "Potential Loyalists" segment
potential_loyalists_segment = filtered_data[filtered_data['Segment'] == 'Potential Loyalists']

# Summary statistics for key metrics related to engagement and retention
potential_loyalists_summary = potential_loyalists_segment[
    ['Total Spend', 'Purchase Frequency', 'Pages Visited', 'Time on Site (min)', 'Promo Response Category']
].describe()

# Count of promo response categories for the Potential Loyalists segment
potential_promo_response_counts = potential_loyalists_segment['Promo Response Category'].value_counts()

potential_loyalists_summary, potential_promo_response_counts


# In[85]:



champions_segment=filtered_data[filtered_data['Segment'] == 'Champions']
champions_segment_detailed = champions_segment[
    ['Total Spend', 'Purchase Frequency', 'Pages Visited', 'Time on Site (min)', 'Promo Response Category']
]

# Quantitative summaries for engagement and spending metrics
champions_spending_engagement = champions_segment_detailed.describe()

# Distribution of promo response categories (quantified)
champions_promo_response_dist = champions_segment_detailed['Promo Response Category'].value_counts(normalize=True) * 100

# Analyzing engagement by promo response category
promo_engagement_analysis = champions_segment.groupby('Promo Response Category')[
    ['Total Spend', 'Pages Visited', 'Time on Site (min)']
].mean()

champions_spending_engagement, champions_promo_response_dist, promo_engagement_analysis


# In[86]:


# Filter data for the "At Risk" segment
at_risk_segment = filtered_data[filtered_data['Segment'] == 'At Risk']

# Summary statistics for key metrics related to engagement and retention
at_risk_summary = at_risk_segment[
    ['Total Spend', 'Purchase Frequency', 'Pages Visited', 'Time on Site (min)', 'Promo Response Category']
].describe()

# Count of promo response categories for the At Risk segment
at_risk_promo_response_counts = at_risk_segment['Promo Response Category'].value_counts()

# Quantify challenges
# 1. Total Spend Quartiles
at_risk_spend_quartiles = at_risk_segment['Total Spend'].quantile([0.25, 0.5, 0.75])
low_spenders_count_at_risk = at_risk_segment[at_risk_segment['Total Spend'] <= at_risk_spend_quartiles[0.25]].shape[0]
total_customers_at_risk = at_risk_segment.shape[0]
low_spenders_percentage_at_risk = (low_spenders_count_at_risk / total_customers_at_risk) * 100

# 2. Engagement Metrics Outliers
low_pages_visitors_count_at_risk = at_risk_segment[at_risk_segment['Pages Visited'] <= 10].shape[0]
low_time_spent_count_at_risk = at_risk_segment[at_risk_segment['Time on Site (min)'] <= 10].shape[0]

# 3. Promo Response Risks
low_promo_response_count_at_risk = at_risk_promo_response_counts.get('Low', 0)
low_promo_response_percentage_at_risk = (low_promo_response_count_at_risk / total_customers_at_risk) * 100

# Generate visualizations
import matplotlib.pyplot as plt

# 1. Distribution of Total Spend
plt.figure(figsize=(8, 3))
at_risk_segment['Total Spend'].plot(kind='hist', bins=20, color='darkblue', edgecolor='black')
plt.title('Distribution of Total Spend (At Risk)')
plt.xlabel('Total Spend ($)')
plt.ylabel('Frequency')
plt.tight_layout()



# 2. Engagement Metrics (Pages Visited and Time on Site)
plt.figure(figsize=(8, 3))
at_risk_segment[['Pages Visited', 'Time on Site (min)']].plot(kind='box', patch_artist=True)
plt.title('Engagement Metrics (At Risk)')
plt.ylabel('Values')
plt.xticks([1, 2], ['Pages Visited', 'Time on Site'])
plt.tight_layout()



# 3. Promo Response Categories
plt.figure(figsize=(8, 3))
at_risk_promo_response_counts.plot(kind='bar', color='darkblue', edgecolor='black')
plt.title('Promo Response Categories (At Risk)')
plt.xlabel('Promo Response Category')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)

# Challenges summary
at_risk_challenges = {
    "Total Customers": total_customers_at_risk,
    "Low Spend Quartile Count": low_spenders_count_at_risk,
    "Low Spend Quartile Percentage": low_spenders_percentage_at_risk,
    "Low Pages Visitors Count": low_pages_visitors_count_at_risk,
    "Low Time Spent Count": low_time_spent_count_at_risk,
    "Low Promo Response Count": low_promo_response_count_at_risk,
    "Low Promo Response Percentage": low_promo_response_percentage_at_risk
}

at_risk_challenges


# In[87]:


# Reloading the dataset to identify "Support Interactions" and "Issues Resolved"

rfm_data_new= filtered_data  # Stripping spaces in column names for accuracy

# Verifying the presence of the necessary columns
if 'Support Interactions' in rfm_data_new.columns and 'Issues Resolved' in rfm_data_new.columns:
    # Perform support-related analysis using these columns
    support_analysis = rfm_data_new.groupby(['Segment', 'Issues Resolved'])[['Total Spend', 'Support Interactions']].mean().reset_index()
    support_analysis.columns = ['Segment', 'Issues Resolved', 'Average Spend', 'Average Support Interactions']

    # Adding likelihood to spend more (simulate impact: +20% for resolved, -10% for unresolved)
    support_analysis['Likelihood Impact'] = support_analysis.apply(
        lambda row: row['Average Spend'] * 1.2 if row['Issues Resolved'] == 'Yes' else row['Average Spend'] * 0.9,
        axis=1
    )

    # Reshape for better clarity
    segment_wise_support = support_analysis.pivot(
        index='Segment',
        columns='Issues Resolved',
        values=['Average Spend', 'Average Support Interactions', 'Likelihood Impact']
    )
    segment_wise_support.columns = [
        'Avg Spend - No Issue Resolved', 'Avg Spend - Issue Resolved',
        'Avg Support - No Issue Resolved', 'Avg Support - Issue Resolved',
        'Impact Spend - No Issue Resolved', 'Impact Spend - Issue Resolved'
    ]

    # Display the results
    (segment_wise_support)

segment_wise_support


# In[88]:


#filtered_data.to_csv( 'C:/Users/anamo/Downloads//rfm_op2.csv')
#segmentation_rfm.to_csv( 'C:/Users/anamo/Downloads//rfm_segmented2.csv')


# In[89]:


segmentation_rfm=filtered_data.groupby(['Segment'])['Total Spend'].sum().reset_index()

segmentation_rfm['Total Spend%']=segmentation_rfm['Total Spend']/sum(segmentation_rfm['Total Spend'])*100
segmentation_rfm=segmentation_rfm.merge((rfm_segments), on='Segment')
segmentation_rfm= segmentation_rfm[['Segment','Customer Count',
       'Distribution', 'Total Spend', 'Total Spend%']]
segmentation_rfm


# In[90]:


segmentation_rfm.columns


# In[91]:


type(rfm_segments)


# In[92]:


# Visualizing the RFM segments
plt.figure(figsize=(4, 3))
rfm_segments.set_index("Segment").plot(
    kind="bar",
    color="blue",
    legend=False,
    figsize=(4, 3),
)
plt.title("Customer Distribution by RFM Segment", fontsize=14)
plt.xlabel("Segment", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[93]:


# Divide Promo Response Rate into 4 classes based on quartiles
sample_data=filtered_data
sample_data['Promo Response Class'] = pd.qcut(
    sample_data['Promo Response Rate'], 
    q=4, 
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Calculate average Promo Response Rate for each Age Group and Promo Response Class
response_analysis = sample_data.groupby(['Age Group', 'Promo Response Class'])['Promo Response Rate'].mean().reset_index()


# ### Insights with Numbers for Retention Strategies:
# 
# #### 1. **Age Group-Specific Engagement:**
#    - **18-25 Age Group:**
#      - **Very High Response Rate:** 87.9%
#      - **High Response Rate:** 65.7%
#      - Suggests strong engagement. Focus on gamified promotions and time-sensitive offers.
#    - **26-35 Age Group:**
#      - **Very High Response Rate:** 85.4%
#      - **High Response Rate:** 63.8%
#      - Indicates good responsiveness. Personalization based on preferences can further increase retention.
#    - **56-65 Age Group:**
#      - **Very High Response Rate:** 72.5%
#      - **High Response Rate:** 55.6%
#      - Lower engagement compared to younger groups. Use value-driven messaging and simplify promotional mechanics.
# 
# #### 2. **Promotion Effectiveness by Response Class:**
#    - **Low Response Class:**
#      - Across all age groups, response rates average **20-25%**.
#      - Indicates significant disengagement. Experiment with new promotional formats, such as free trials or educational offers.
#    - **Very High Response Class:**
#      - Average response rates across age groups exceed **75%**.
#      - These customers are highly engaged and should receive exclusive perks like early access to sales, loyalty benefits, or personalized offers.
# 
# #### 3. **Age Group and Strategy Suggestions:**
#    - **18-25 Age Group:**
#      - Average promo response for "Very High" is **87.9%**. Leverage social media, influencer campaigns, and mobile-based interactions.
#    - **26-35 Age Group:**
#      - Average response for "High" is **63.8%**, and for "Very High" is **85.4%**. Personalization based on shopping patterns will work well.
#    - **46-55 Age Group:**
#      - Average response for "Medium" is **40.1%**, which is lower than younger segments. Highlight practical value through cashback or bundled offers.
#    - **66+ Age Group:**
#      - Average response for "Low" is **20.4%**. Traditional promotional methods, such as email and direct mail, can improve engagement.
# 
# #### 4. **Retention Focus by Class:**
#    - **Highly Responsive Customers (High and Very High):**
#      - Across all age groups, these segments average **75%+** response rates. Offer early access to promotions and personalized experiences.
#    - **Moderately Responsive Customers (Medium):**
#      - Response rates range from **40-50%** across most age groups. Retarget with A/B-tested offers to identify effective strategies.
#    - **Low Responders:**
#      - Response rates average **20-25%**. Conduct surveys or personalized outreach to understand preferences.
# 
# ### Summary:
# These numbers clearly show that younger age groups (18-35) have higher promo response rates, particularly in the "High" and "Very High" classes. Older age groups (56+) show lower engagement, emphasizing the need for age-appropriate promotional strategies. Tailoring promotions to specific segments based on their engagement levels and age can significantly boost retention. 
# 
