import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
abr_month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
            'aug', 'sep', 'oct', 'nov', 'dec']


def load_monthly_data():
    """Loads monthly sales data from  csv"""

    monthly = pd.read_csv(
        'SELL_1.csv',
        delimiter=';',
        decimal=',',
        encoding='latin-1')
    monthly = monthly.rename(columns={
        'Date': 'date',
        'PKod': 'sku',
        'Pgroup': 'group',
        'Pname': 'name',
        'Pquantity': 'quantity',
        'pce_zn': 'unit_cogs',
        'pwa_zn': 'agg_cogs',  # aggregate cogs for this sku
        'pce_sn': 'unit_revenue',
        'pwn_sn': 'agg_rev',  # aggregate revenue for this sku
        'pmarza': 'gross_margin',
        'pmarzajedn': 'unit_contribution',
        'pkwmarza': 'agg_dollar_contribution',
        'pudzmarza': 'share_of_margin'
    })

    monthly = monthly.drop(['pwa_sn', 'pce_sb', 'pwa_sb', 'pudzsb'], axis=1)
    monthly.group = monthly.group.str.lower()
    monthly.name = monthly.name.str.lower()
    monthly.date = pd.to_datetime(monthly.date, format="%d.%m.%Y")
    monthly.unit_cogs = monthly.unit_cogs.str.replace(
        ',', '.').str.replace(
        ' ', '')

    monthly.group = monthly.group.replace(
        {"ketch_concetrate_mustard_majo_horseradish": "sauce"})

    return monthly

monthly = load_monthly_data()
print(monthly.head())
# monthly = load_monthly_data()

# Drop empty rows
idx_to_drop = monthly.unit_cogs[monthly.unit_cogs.str.len() == 0].index
monthly = monthly.drop(idx_to_drop)

# Convert to numeric
monthly.unit_cogs =  monthly.unit_cogs.astype('float')

# Add aggregate revenue column
monthly['agg_rev'] = monthly['quantity'] * monthly['unit_revenue']
monthly.isnull().sum()
monthly.duplicated().sum()
# print(monthly.columns)
categorical_cols = ['group']
numeric_cols = list(monthly.columns)
numeric_cols.remove('group')
monthly.describe(include = ['O'])
monthly.describe(include = [np.number])
print(monthly.head())

# Getting the full list of categories and the number of skus within each category
category_sku_count = monthly[['sku','group']].drop_duplicates().groupby('group').count().sort_values(by='sku', ascending=False)
category_annual_contribution = monthly[['group','agg_dollar_contribution']].drop_duplicates().groupby('group').sum().sort_values(by='agg_dollar_contribution', ascending=False)
category_sku_annual_contrib = category_sku_count.join(category_annual_contribution)

# calculating aggregate dollar revenue
monthly['agg_revenue'] = monthly['quantity'] * monthly['unit_revenue']
print(monthly.head())
labels = list(category_sku_annual_contrib.index)
x = np.arange(len(labels))
width = .35

fig, ax1 = plt.subplots(figsize=(20,6))
ax1.bar(x - width/2, category_sku_count.sku, width, color=['orange'])
ax2 = ax1.twinx()
ax2.bar(x+ width/2, category_annual_contribution.agg_dollar_contribution, width)

ax1.legend(['SKU'])
ax2.legend(['Contribution'],loc=2)

ax1.set_ylabel('# of SKUs')
ax2.set_ylabel('Aggregate Annual Contribution (PLN)')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, minor=False, rotation=90);
plt.show(ax1)
# plt.waitforbuttonpress(0)


def get_mom_growth(df):
    """Get the month over month growth rate for input data"""

    growth_df = df / df.shift(1, axis=1) - 1
    return growth_df.iloc[:, 1:]


def top_n_cat_by_contrib_idx(n):
    """Returns sorted n largest product categories index (pd.Int64Index) by annual aggregate revenue"""

    return category_contribution_by_month.sum(axis=1).sort_values(ascending=False).iloc[:n].index


def ma(df, n):
    """Returns a moving average dataframe based on given period"""

    return df.rolling(window=n, axis=1).mean().iloc[:, n - 1:]


def cat_by_month(value, aggfunc='sum'):
    """Returns df aggregated on value (fn parameter). Returned df has product categories as rows
    and month of year as columns"""

    df = monthly.pivot_table(value, index='group', columns=['date'], aggfunc=aggfunc)
    df = df.fillna(0)
    df.columns = abr_month
    return df


# These functions have some dependent dataframes
category_contribution_by_month = cat_by_month('agg_revenue') - cat_by_month('agg_cogs')

fig1, axs = plt.subplots(4, 4, figsize=(18, 18), sharex=False)
i = 0

monthly_items_sold_by_category = cat_by_month('sku', 'count').T
monthly_items_sold_by_category = monthly_items_sold_by_category[list(top_n_cat_by_contrib_idx(17))]

categories_ex_vegetables = list(monthly_items_sold_by_category.columns.values)
categories_ex_vegetables.remove('vegetables')

# Plotting the subplots
for row in range(9):
    for col in range(4):
        try:
            cat = categories_ex_vegetables[i]
            i += 1

            x = monthly_items_sold_by_category.vegetables.values
            y = monthly_items_sold_by_category[cat].values
            sns.regplot(x=x, y=y, ax=axs[row][col])
            axs[row][col].title.set_text(cat)

            if col == 0:
                axs[row][col].set_ylabel('# of Items Sold')


            if row == 3:
                axs[row][col].set_xlabel('# of Grocery Items Sold')
                # plt.waitforbuttonpress(0)
        except IndexError:
            # Done
            break
plt.show()
# plt.waitforbuttonpress(0)
fig = plt.figure(figsize=(12, 9))

product_categories = list(monthly_items_sold_by_category.columns.values)
# sns.heatmap(monthly_items_sold_by_category[list(top_n_cat_by_contrib_idx(10))].corr(),
#             cmap=sns.color_palette("RdBu_r",50)[::-1],
#             annot=True)

# plt.xticks(rotation=45)
# plt.show()

# Making the sales correlation matrix between groceries and other categories
corr_df = pd.DataFrame(np.corrcoef(monthly_items_sold_by_category.values, rowvar=False), columns = monthly_items_sold_by_category.columns)
corr_df.index = monthly_items_sold_by_category.columns
corr_df = corr_df[['vegetables']].sort_values('vegetables', ascending=False)

print(corr_df.head())

# Aggregate dollar contribution by category
contribution_by_category = monthly[['group','agg_dollar_contribution']].groupby(by='group').sum()
contribution_by_category = contribution_by_category.sort_values(by='agg_dollar_contribution', ascending=False)
contribution_by_category = contribution_by_category.join(corr_df)
contribution_by_category = contribution_by_category.rename(columns={'vegetables':'corr_w_vegetables'})

# Maximum estimated financial impact by counting everything correlated with grocery sales as sales lift
contribution_by_category['max_est_fin_impact'] = contribution_by_category['agg_dollar_contribution'] * contribution_by_category['corr_w_vegetables']

# Measuring the category's importance by calculating its share of total store's contribution pool
contribution_by_category['cat_total_contrib_share'] =  contribution_by_category['agg_dollar_contribution'] / contribution_by_category['agg_dollar_contribution'].sum()

# Contribution counted as sales lift as a percentage of total store's contribution
contribution_by_category['lift_contrib_share'] =  contribution_by_category['max_est_fin_impact'] / contribution_by_category['agg_dollar_contribution'].sum()

max_lift = contribution_by_category.loc[contribution_by_category['lift_contrib_share'] > 0, 'lift_contrib_share'].sum()

print("Max estimated lift is {0:.1f}% counting fresh produce sales, and {1:.1f}% without.".format(max_lift * 100, (max_lift - .1) * 100))
# Looking at the dataframe containing sales lift data
print(contribution_by_category.sort_values('max_est_fin_impact', ascending=False).head())

# Grabbing data for skus sold and revenue per month
monthly_aggregate_skus_sold = cat_by_month('quantity').sum(axis=0)
monthly_aggregate_revenue = cat_by_month('agg_revenue').sum(axis=0)
monthly_aggregate_revenue.index = monthly_aggregate_skus_sold.index

# plotting side by side
rev_sku_fig = plt.figure(figsize=(12,6))
ax1 = rev_sku_fig.add_subplot(111)
ax1.plot(monthly_aggregate_skus_sold)
ax1.set_ylabel('# of Items Sold')

ax2 = ax1.twinx()
ax2.plot(monthly_aggregate_revenue, 'r-')
ax2.set_ylabel('Monthly Revenue (PLN)')

ax1.set_title("# of Items Sold and Monthly Revenue")

ax1.legend(['Items Sold'])
ax2.legend(['Monthly Rev'],loc=2)

for tl in ax2.get_yticklabels():
    tl.set_color('r')

plt.show(ax2)
# plt.waitforbuttonpress(0)
# Products, a df conaining SKU, product group and product name
products = monthly[['group','sku','name']].drop_duplicates().set_index('sku')
print(products.head())

monthly_sku_revenue = monthly[['date','sku','agg_revenue']].pivot_table('agg_revenue', index='sku', columns='date')
monthly_sku_revenue = monthly_sku_revenue.fillna(0)
monthly_sku_revenue.columns = abr_month
monthly_sku_revenue['annual_total'] = monthly_sku_revenue.sum(axis=1)
monthly_sku_revenue = monthly_sku_revenue.sort_values('annual_total', ascending=False)

# Cumulative Revenue by nth Top Performing SKUs¶
# Calculating each SKU's share of overall revenue
product_performance = monthly_sku_revenue.join(products)
product_performance['rev_share'] = product_performance['annual_total'] / product_performance['annual_total'].sum()

# Sort by most performent SKU based on annual contribution and calculating cumulative revenue
product_performance['cumulative_rev'] = product_performance['rev_share']

cumulative_revenue_share = 0.0
for idx, row in product_performance.iterrows():
    product_performance.loc[idx, 'cumulative_rev'] = cumulative_revenue_share + row['rev_share']
    cumulative_revenue_share += row['rev_share']

# Making a pricing table to hold product pricing information
pricing = monthly[['sku', 'unit_cogs', 'unit_revenue']]
# In case there are multiple prices for a single SKU, we'll take the mean
pricing = pricing.groupby('sku').mean()

# Joining dfs together
product_performance = product_performance.join(pricing)
print(f'there are {product_performance.shape[0]} SKUs sold by the store within the last yr, \
but many also has not made a sale in months')
print(product_performance.head())

# Plotting Cumulative Revenue
cumulative_revenue = pd.DataFrame(product_performance['cumulative_rev'].values)

plt.figure(figsize=(9,5))
plt.plot(cumulative_revenue.values)
plt.title('Cumulative Revenue by nth Top SKUs')
plt.xlabel("n'th SKUs")
plt.ylabel("Cumulative Revenue as % of Total");
plt.show()


def get_sku_status(df, n_months, active=True):
    """Returns the index of active or inactive SKUs. A inactive SKU is one that has not made a
    sale in the last n_months.

    Args:
        df: df
            product dataframe containing product sku, cumulative revenue, unit cogs, and unit revenue

        n_months: int
            A SKU that has not made a sale in n_months will be marked as inactive

        active: bool
            Returns active SKU indexes when set to True, returns inactive SKU indexes when False


    Returns:
        idx: pd.Int64Index
            Pandas Row Index of active or inactive SKUs
    """
    start_month = abr_month[-n_months]
    df['rev_last_n_months'] = df.loc[:, start_month:"dec"].sum(axis=1)

    if active:
        return df[df['rev_last_n_months'] > 0].index

    else:
        return df[df['rev_last_n_months'] <= 0].index


def working_capital_used_by_underperfoming_products(df, rev_to_keep, liquidation_discount, inventory_per_sku):
    """Calculate the amount of working capital used by non-performent SKUs, and the amount of capital
    that can be released from liduidating these non-performent SKUs. You can specify the amount of
    revenues you wish to retain. (IE, find the long tail SKUs that constitues the latst 10% of revenue)

    Args:
        df: df
            product dataframe containing product sku, culative revenue, unit cogs, and unit revenue

        rev_to_keep: float
            Amount of revenue to *keep*, ranges from 0 to 1, 1 being keeping all revenue and not
            dropping any skus. Similarily, 0 being keeping no revenue and dropping all SKUs

        liquidation_discount: float
            A discount (0 to 1) applied to retail price to clear inventory. It is used to calcualte
            amount of capital that can be freed up via inventory liquidation

    Returns:
        working_cap_invested: float
            Amount of working capital invested in non-performent SKUs

        working_cap_freed: float
            Amount of working capital to be freed by liquidating stock at liquidation discount

        num_skus_dropped: int
            Number of SKUs dropped

        idx_skus_dropped: df.Int64Index
            dataframe index of SKUs that will be dropped

    """
    # Get a list of active SKUs (SKU that made sales in last 3 months)
    active_skus_idx = get_sku_status(df, 3, active=True)
    df = df.loc[active_skus_idx, :].copy()

    active_skus = df.shape[0]

    # Calculating working capital and liquidation value
    df['working_cap'] = df['unit_cogs'] * inventory_per_sku
    df['liquidation_cf'] = df['unit_revenue'] * liquidation_discount * inventory_per_sku

    # return totals for given revene share to keep
    df_to_liquidate = df[df['cumulative_rev'] >= rev_to_keep]
    total_wc = df_to_liquidate['working_cap'].sum()
    total_liquidation_cf = df_to_liquidate['liquidation_cf'].sum()

    num_skus_dropped = df_to_liquidate.shape[0]
    num_skus_dropped_frac = num_skus_dropped / active_skus * 100

    idx_skus_dropped = df_to_liquidate.index

    print("Total working capital invested in SKUs to be liquidated is ${0:.0f} PLN".format(total_wc))
    print("Total amount of cash generated from liquidating underperforming SKUs is ${0:.0f} PLN".format(
        total_liquidation_cf))
    print("Total number of active SKUs to be dropped from store is {0:.0f}, representing {1:.0f}% of your SKUs".format(
        num_skus_dropped, num_skus_dropped_frac))

    return total_wc, total_liquidation_cf, num_skus_dropped, idx_skus_dropped

_, _, _, idx_skus_dropped = working_capital_used_by_underperfoming_products(product_performance, .9, 0.35, 5);
def explore_dropped_sku_categories(product_performance, idx_skus_dropped, n_months):
    """
    Calculates number of SKUs to be dropped in each product category and compare it to the total
    number of SKUs in that produce category

    Args:
        product_performance: df
            product dataframe containing product sku, culative revenue, unit cogs, and unit revenue

        idx_skus_dropped: pd.Int64Index
            index conaining skus to be dropped

        n_months: int
            Threshold in months used in marking a item as inactive. Every SKU that has made
            a sale in the last n_months are considered active, and as such, is assumed to be
            currently in inventory. Active SKUs are counted towards category SKU total. Inactive
            SKUs are not.

    """
    skus_dropped = product_performance.loc[idx_skus_dropped, :].groupby('group').count()
    skus_dropped = pd.DataFrame(skus_dropped.name.sort_values(ascending=False))
    skus_dropped.columns = ['skus_dropped']

    active_skus_idx = get_sku_status(product_performance, 3, active=True)
    active_skus = product_performance.loc[active_skus_idx, :]

    # Count the categories
    active_skus_per_category = active_skus.groupby('group').count()
    active_skus_per_category = pd.DataFrame(active_skus_per_category.name)
    active_skus_per_category.columns = ['cat_total']

    # Join the dfs and normalize
    df = skus_dropped.join(active_skus_per_category)
    df['frac_dropped'] = df.skus_dropped / df.cat_total

    return df

print(explore_dropped_sku_categories(product_performance, idx_skus_dropped, 3).sort_values('frac_dropped', ascending=False))
store_by_month = monthly[['date','quantity','agg_rev', 'agg_dollar_contribution']].groupby('date').sum()
store_by_month['gross_margin'] = store_by_month['agg_dollar_contribution'] / store_by_month['agg_rev']
print(store_by_month.head())

# Loser look at margins
store_by_month['gross_margin'].plot()
plt.ylabel('Margin')
plt.title("Store Wide Margin by Month");

# Contribution magin for each category, broek down by months
# Contribution margin = 1 - aggregate_cogs / aggregate_revenue
category_contribution_margin_by_month = 1 - cat_by_month('agg_cogs') / cat_by_month('agg_revenue')

# Store baseline performance
store_baseline_margin = 1 - cat_by_month('agg_cogs').sum() / cat_by_month('agg_revenue').sum()

print(category_contribution_margin_by_month.head())

# We want to subtract category growth by baseline store level performance to see which category is
# Over performing, and which category is underperforming relative to baseline.

# Pandas doesn't support broadcasting so we'll make the df ourselves
base_row = pd.DataFrame(store_baseline_margin).T
to_append = []
for idx in list(category_contribution_margin_by_month.index):
    to_append.append(base_row.rename({0:idx}))

broadcasted_baseline_margin = pd.concat(to_append)

# Month over month margin change
category_margin_performance = get_mom_growth(category_contribution_margin_by_month)
category_margin_performance = category_margin_performance.sort_values('jun', ascending=False).loc[top_n_cat_by_contrib_idx(8),:]

# Month over month margin change over baseline (store wide m/m margin change)
category_margin_performance_over_baseline = get_mom_growth(category_contribution_margin_by_month) - get_mom_growth(broadcasted_baseline_margin)
category_margin_performance_over_baseline = category_margin_performance_over_baseline.sort_values('jun', ascending=False).loc[top_n_cat_by_contrib_idx(7),:]

# Using 3 month moving average to smooth out the volatility a bit more and make the trend more appearant

fig, ax = plt.subplots(1, figsize=(12,6))
ma(category_margin_performance_over_baseline, 3).T.plot(ax=ax)
ax.set_title("M/M Category Contribution Margin Change, 3 Month Moving Average")
ax.grid(axis='y')
ax.set_xticklabels(category_margin_performance.T.index[1:]);
plt.show(ax)

# Seems like the positive margin expansion has been casued by the margin expansion of fresh produce
# Explore more by looking at volume change
category_quantity_performance = get_mom_growth(cat_by_month('quantity')).loc[top_n_cat_by_contrib_idx(8),:]

fig, ax = plt.subplots(1, figsize=(12,6))
ma(category_quantity_performance, 3).T.plot(ax=ax)
ax.set_title("M/M Category Volume Change, 3 Month Moving Average")
ax.grid(axis='y')
ax.set_xticklabels(category_quantity_performance.T.index[1:]);
plt.show(ax)


def get_best_product_for_quarter(q, top_n=200):
    """Get a list of best performing products for a given quarter

    Args:
        q: int
            Quarter of the year, 1 to 4
            1: Start of Jan - end of Mar
            2: Start of Apr - end of Jun
            3: Start of Jul - end of Sept
            4: Start of Oct - end of Dec

        top_n: int
            Get up to top nth SKU

    Returns:
        df: DataFrame
            df containg best performing SKUs
    """
    df = product_performance.copy()
    df['q1_contrib'] = df.loc[:, 'jan':'mar'].sum(axis=1)
    df['q2_contrib'] = df.loc[:, 'apr':'jun'].sum(axis=1)
    df['q3_contrib'] = df.loc[:, 'jul':'sep'].sum(axis=1)
    df['q4_contrib'] = df.loc[:, 'oct':'dec'].sum(axis=1)

    quarters = {1: 'q1_contrib', 2: 'q2_contrib', 3: 'q3_contrib', 4: 'q4_contrib'}

    quarter = quarters[q]

    return df.sort_values(quarter, ascending=False).iloc[:top_n, :]

print(get_best_product_for_quarter(1).head())

# What categories do top SKUs in Q1 belong to?
# If you do this for all 4 quarters you will see these rankings stays relatively constant,
# meaning all categories exhibit similar seasonality trends
get_best_product_for_quarter(1).group.value_counts().head(7)


# Getting a list of products that have significant sales. This will make sure we have meaningful month over month
# growth metrhics. We'll use 1500 PLN annual sales as the bar

def get_highest_growth_skus(top_n, min_annual_pln, month):
    """Returns a dataframe with highest growth skus

    Args:
        top_n: int
            Get top n high growth skus

        min_annual_pln: int
            Minimum annual sales in PLN the sku needs to make in order for it to be considered as a potential
            high growth candidate

        month: str
            Abbrevated month to rank the high growth skus

    Returns:
        df: DataFrame
            df containing high growth skus sorted by the month of choice, values in df are 2 month
            moving average of dollar contribution growth

    """
    significant_skus_idx = product_performance[product_performance.annual_total > min_annual_pln].index

    # High Growth Products
    high_growth_skus = ma(get_mom_growth(product_performance.copy().loc[:, 'jan':'dec']), 2)

    high_growth_skus = high_growth_skus.loc[significant_skus_idx, :].sort_values(month, ascending=False)

    high_growth_skus = high_growth_skus.join(product_performance[['group', 'name']])

    return high_growth_skus


# Getting a list of products that have significant sales. This will make sure we have meaningful month over month
# growth metrhics. We'll use 1500 PLN annual sales as the bar

def get_highest_growth_skus(top_n, min_annual_pln, month):
    """Returns a dataframe with highest growth skus

    Args:
        top_n: int
            Get top n high growth skus

        min_annual_pln: int
            Minimum annual sales in PLN the sku needs to make in order for it to be considered as a potential
            high growth candidate

        month: str
            Abbrevated month to rank the high growth skus

    Returns:
        df: DataFrame
            df containing high growth skus sorted by the month of choice, values in df are 2 month
            moving average of dollar contribution growth

    """
    significant_skus_idx = product_performance[product_performance.annual_total > min_annual_pln].index

    # High Growth Products
    high_growth_skus = ma(get_mom_growth(product_performance.copy().loc[:, 'jan':'dec']), 2)

    high_growth_skus = high_growth_skus.loc[significant_skus_idx, :].sort_values(month, ascending=False)

    high_growth_skus = high_growth_skus.join(product_performance[['group', 'name']])

    return high_growth_skus

# Getting the highest growth SKUs for jun, next year this time you can promote these a bit more, put them on
# better shelf locations etc...

print(get_highest_growth_skus(50, 18, 'jun').head())

print()

