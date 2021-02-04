#get all packages we need
import pandas as pd
import math
import scipy.stats
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

#read the data
#download data here: https://drive.google.com/drive/folders/10Q7OSn4BmeW6aoQy1AvvjcEtI3EC9ffW?usp=sharing
df_test_overall = pd.read_csv('./visit-data.csv')
df_cateforymap = pd.read_csv('./category-mapping.csv')
df_rev = pd.read_csv('./rev-data.csv')

#pull the data we need (test2)
df_test = df_test_overall[df_test_overall['testid'] == 2]
df_test = df_test.merge(df_cateforymap, how = 'left', on = 'ChannelID')
df_test['Channel Name'] = df_test['Channel Name'].fillna('Undefined')
df_test.drop_duplicates(['Date', 'SessionID']).groupby(['Date'])['SessionID'].count()
df_test = df_test[pd.to_datetime(df_test['Date']) > pd.to_datetime('2019-06-15')]
df_test = df_test[pd.to_datetime(df_test['Date']) < pd.to_datetime('2019-07-12')]
df_test

#get revenue data ready as array format for removeing outliers
df_rev = df_rev[df_rev['testid'] == 2]
control_rev = df_rev[df_rev['ControlGroup'] == 1]['TotalRevenue'].array
var_rev = df_rev[df_rev['ControlGroup'] == 0]['TotalRevenue'].array

#plot data and find out outliers
x = df_rev.loc[df_rev['ControlGroup'] == 0, 'TotalRevenue'].apply(lambda x: math.log(x, 2)) #算出x是2的几次方

n_bins = 50

fig,ax = plt.subplots(figsize = (16,8)) 

n, bins, patches = ax.hist(x, n_bins, density = True, histtype = 'step', cumulative=True, label='Cumulative')

ax.grid(True)
ax.legend(loc = 'right')
ax.set_title('Cumulative step histograms')
ax.set_xlabel('Total rev')
ax.set_ylabel('Likelihood of occurrence')

#set scale interval as 0.05 for the plot
y_major_locator=MultipleLocator(0.05)
ax.yaxis.set_major_locator(y_major_locator)

plt.show()

#according to the chart, get the interval
p1 = np.percentile(control_rev, [5, 95])
p2 = np.percentile(var_rev, [5, 95])
p1
p2

#add TotalRevenue_sq column for df_reft
df_rev['TotalRevenue_sq'] = df_rev['TotalRevenue']**2

#respectively remove outlier for control group(1) and test group(0) by using the interval we got from previous step(p1 & p2)
df_rev_1 = df_rev[(df_rev['TotalRevenue'] > 36.7825) & (df_rev['TotalRevenue'] < 990) & (df_rev['ControlGroup'] == 1)]
df_rev_0 = df_rev[(df_rev['TotalRevenue'] > 36.42) & (df_rev['TotalRevenue'] < 985.929) & (df_rev['ControlGroup'] == 0)]

#merge df_rev_1 & df_rev_0 without outliers as df_rev_new
df_rev_new = pd.concat([df_rev_1, df_rev_0], axis = 0)
df_rev_new = df_rev_new.rename(columns = {'"Date"': 'Date'})

#merge df_test and df_rev_new
df_test = df_test.merge(df_rev_new, how = 'left', on = ['SessionID', 'Date', 'ControlGroup', 'testid'])
df_test['TotalRevenue'] = df_test['TotalRevenue'].fillna(0)
df_test['TotalRevenue_sq'] = df_test['TotalRevenue_sq'].fillna(0)

#overwrite revenue data as array format again for the following mann whitney method
control_rev = df_test[df_test['ControlGroup'] == 1]['TotalRevenue'].array
var_rev = df_test[df_test['ControlGroup'] == 0]['TotalRevenue'].array


# # A/B Test (Continuous)
#method A：simply get all metrics we need by using scipy.stats.mannwhitneyu function, however this method is less strict, easier to reach significance level
scipy.stats.mannwhitneyu(control_rev, var_rev)

#method B:
#define z test calculator for continuous distribution 
def z_test_calculator_continuous (df, denominator, numerator, numerator_sq):
    
    #get data
    control_denominator = df.loc[1, denominator]
    var_denominator = df.loc[0, denominator]
    
    control_numerator = df.loc[1, numerator]
    var_numerator = df.loc[0, numerator]
    
    #get rate
    control_rate = control_numerator / control_denominator
    var_rate = var_numerator / var_denominator
    
    #variance sq
    control_var = df.loc[1, numerator_sq] / control_denominator - control_rate**2
    var_var = df.loc[0, numerator_sq] / var_denominator - var_rate**2
    
    #z score
    z_score = (control_rate - var_rate) / math.sqrt(control_var / control_denominator + var_var / var_denominator)
    
    #p value
    p_value = scipy.stats.norm.sf(abs(z_score)) 
    
    #lift
    perc_lift = (var_rate - control_rate) / control_rate
    abs_lift = var_rate - control_rate
    
    return (p_value, perc_lift, abs_lift)


#define ab test calculator for continuous distribution(revenue)
def ab_test_calculator_rev (df, data_type): #df = df_test, data_type = ['SessionID', 'CusID']
    
    #integrate all necesary input data into df_result by using loop
    metrics = ['SawProduct', 'Bounced', 'AddedToCart', 'ReachedCheckout', 'Converted']

    df_result_rev = df.drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[[data_type]].count()

    for metric in metrics:
        a = df[df[metric] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[data_type].count()
        a.name = metric
        df_result_rev = df_result_rev.join(a)
        
    df_result_rev.loc[1, 'Rev'] = sum(df[df['ControlGroup'] == 1]['TotalRevenue'])
    df_result_rev.loc[1, 'Rev_sq'] = sum(df[df['ControlGroup'] == 1]['TotalRevenue_sq'])

    df_result_rev.loc[0, 'Rev'] = sum(df[df['ControlGroup'] == 0]['TotalRevenue'])
    df_result_rev.loc[0, 'Rev_sq'] = sum(df[df['ControlGroup'] == 0]['TotalRevenue_sq'])
    
    dic_final_rev = {}
    df_final_rev =pd.DataFrame() 

    #input specific KPIs to get coresponding outputs by using z_test_calculator & fill in all outputs in df_final
    KPIs = [data_type, 'SawProduct', 'AddedToCart', 'ReachedCheckout', 'Converted']

    j = 0

    for KPI in KPIs:
        result = z_test_calculator_continuous(df_result_rev, KPI, 'Rev', 'Rev_sq')
        df_final_rev.loc[j, 'denominator'] = KPI
        df_final_rev.loc[j, 'numerator'] = 'Rev'
        df_final_rev.loc[j, 'p_value'] = result[0]
        df_final_rev.loc[j, 'perc_lift'] = result[1]
        df_final_rev.loc[j, 'abs_lift'] = result[2]
        j = j + 1

        dic_final_rev['TestGroup'] = df_final_rev
        
    return df_final_rev


#define reference line tool to realize automation
def reference_line_rev(df, denominator, data_type): #df = df_test, denominator = ['SessionID'~'Converted'], data_type = ['SessionID', 'CusID']
    
    #get daily data for denominator
    if denominator == data_type:
        denominator_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count()

    elif denominator != data_type:    
        denominator_dailydata = df[df[denominator] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count()
    
    #get daily data for numerator
    if data_type == 'CusID':
        numerator_dailydata = df.drop_duplicates([data_type, 'SessionID', 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue'].sum()
        numerator_sq_dailydata = df.drop_duplicates([data_type, 'SessionID', 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue_sq'].sum()

    elif data_type != 'CusID':   
        numerator_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue'].sum()
        numerator_sq_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue_sq'].sum()

    #name them
    denominator_dailydata.name = denominator
    numerator_dailydata.name = 'rev'
    numerator_sq_dailydata.name = 'rev_sq'

    #reset index (the original index is 'Date')
    denominator_aggdailydata = denominator_dailydata.reset_index()
    numerator_aggdailydata = numerator_dailydata.reset_index()
    numerator_sq_aggdailydata = numerator_sq_dailydata.reset_index()

    #replace denominator/numearator_aggdailydata's original denominator/ numerator column data with cumsum data
    denominator_aggdailydata[denominator] = denominator_aggdailydata.groupby('ControlGroup')[denominator].cumsum()
    numerator_aggdailydata['rev'] = numerator_aggdailydata.groupby('ControlGroup')['rev'].cumsum()
    numerator_sq_aggdailydata['rev_sq'] = numerator_sq_aggdailydata.groupby('ControlGroup')['rev_sq'].cumsum()

    #calculate p value, perc lift & abs lift for every day by using loop and z test calculator, and then fill them in df_cumsum
    df_cumsum = pd.DataFrame()

    for date in df.drop_duplicates('Date')['Date'].tolist():
        df_result = denominator_aggdailydata[denominator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True)
        df_result = df_result.merge(numerator_aggdailydata[numerator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True), on = ['Date', 'ControlGroup'])
        df_result = df_result.merge(numerator_sq_aggdailydata[numerator_sq_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True), on = ['Date', 'ControlGroup'])

        sig_result = z_test_calculator_continuous(df_result, denominator, 'rev', 'rev_sq')
        df_cumsum.loc[date, 'sig_level'] = 1 - sig_result[0]
        df_cumsum.loc[date, 'perc_lift'] = sig_result[1]
        df_cumsum.loc[date, 'abs_lift'] =  sig_result[2]

    #tidy up the order
    df_cumsum.sort_index()

    #plot
    ax = df_cumsum.sort_index()['sig_level'].plot(figsize = (15, 5))
    ax.axhline(y = 0.95, linewidth = 1, color = 'r')
    y_major_locator = MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.grid(True)

    return ax


# # A/B Test (Continuous) w/ cut 
#from here we define a series of a/b test w/ cut automated functions in similar logic as the above
#define ab test calculator applied to cut
def ab_test_calculator_rev_cut (df, data_type, cut): #df = df_test, data_type = ['SessionID', 'CusID'], cut = ['CategoryID', 'VisitorType', 'Channel Name']
    
    dic_final_rev = {}
    
    df_cut = df.copy()

    #use loop to get results cut by cut
    for p in set(df_cut[cut]):
        
        #integrate all necesary input data into df_result by using loop
        df = df_cut[df_cut[cut] == p]
    	
        metrics = ['SawProduct', 'AddedToCart', 'ReachedCheckout', 'Converted']

        df_result_rev = df.drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[[data_type]].count()

        for metric in metrics:
            a = df[df[metric] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[data_type].count()
            a.name = metric
            df_result_rev = df_result_rev.join(a)
        
        df_result_rev.loc[1, 'Rev'] = sum(df[df['ControlGroup'] == 1]['TotalRevenue'])
        df_result_rev.loc[1, 'Rev_sq'] = sum(df[df['ControlGroup'] == 1]['TotalRevenue_sq'])

        df_result_rev.loc[0, 'Rev'] = sum(df[df['ControlGroup'] == 0]['TotalRevenue'])
        df_result_rev.loc[0, 'Rev_sq'] = sum(df[df['ControlGroup'] == 0]['TotalRevenue_sq'])

        #input specific KPIs to get coresponding outputs by using z_test_calculator_continuous & fill in all outputs in df_final & dic_final
        KPIs = [data_type, 'SawProduct', 'AddedToCart', 'ReachedCheckout', 'Converted']
        
        df_final_rev =pd.DataFrame() 
        
        j = 0

        for KPI in KPIs:
            result = z_test_calculator_continuous(df_result_rev, KPI, 'Rev', 'Rev_sq')
            df_final_rev.loc[j, 'denominator'] = KPI
            df_final_rev.loc[j, 'numerator'] = 'Rev'
            df_final_rev.loc[j, 'p_value'] = result[0]
            df_final_rev.loc[j, 'perc_lift'] = result[1]
            df_final_rev.loc[j, 'abs_lift'] = result[2]
            j = j + 1

            dic_final_rev[cut + '_' + str(p)] = df_final_rev
        
    return dic_final_rev

#define reference line tool applied to cut
def reference_line_rev_cut(df, denominator, data_type, cut): #df = df_test, denominator = ['SawProduct'~'Converted'], data_type = ['SessionID', 'CusID'], cut = ['CategoryID', 'VisitorType', 'Channel Name']
    
    df_copy = df.copy()
    
    dic_cumsum = {}
    
    j = 0
    
    fig, ax = plt.subplots(len(set(df[cut])))
    
    #use loop to get results cut by cut
    for p in set(df_copy[cut]):
        
        df = df_copy[df_copy[cut] == p]
    	
    	#get daily data for denominator
        if denominator == data_type:
            denominator_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()

        elif denominator != data_type:    
            denominator_dailydata = df[df[denominator] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()

        #get daily data for numerator
        if data_type == 'CusID':
            numerator_dailydata = df.drop_duplicates([data_type, 'SessionID', 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue'].sum()
            numerator_sq_dailydata = df.drop_duplicates([data_type, 'SessionID', 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue_sq'].sum()

        elif data_type != 'CusID':   
            numerator_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue'].sum()
            numerator_sq_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])['TotalRevenue_sq'].sum()

        #name them
        denominator_dailydata.name = denominator
        numerator_dailydata.name = 'rev'
        numerator_sq_dailydata.name = 'rev_sq'

        #reset index (the original index is 'Date')
        denominator_aggdailydata = denominator_dailydata.reset_index()
        numerator_aggdailydata = numerator_dailydata.reset_index()
        numerator_sq_aggdailydata = numerator_sq_dailydata.reset_index()

        #replace denominator/numearator_aggdailydata's original denominator/ numerator column data with cumsum data
        denominator_aggdailydata[denominator] = denominator_aggdailydata.groupby('ControlGroup')[denominator].cumsum()
        numerator_aggdailydata['rev'] = numerator_aggdailydata.groupby('ControlGroup')['rev'].cumsum()
        numerator_sq_aggdailydata['rev_sq'] = numerator_sq_aggdailydata.groupby('ControlGroup')['rev_sq'].cumsum()

        #calculate p value, perc lift & abs lift for every day by using loop and z_test_calculator_continuous, and then fill them in df_cumsum
        df_cumsum = pd.DataFrame()

        for date in df.drop_duplicates('Date')['Date'].tolist():
            df_result = denominator_aggdailydata[denominator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True)
            df_result = df_result.merge(numerator_aggdailydata[numerator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True), on = ['Date', 'ControlGroup'])
            df_result = df_result.merge(numerator_sq_aggdailydata[numerator_sq_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True), on = ['Date', 'ControlGroup'])

            sig_result = z_test_calculator_continuous(df_result, denominator, 'rev', 'rev_sq')
            df_cumsum.loc[date, 'sig_level'] = 1 - sig_result[0]

        dic_cumsum['TestGroup_Cut' + str(p)] = df_cumsum.sort_index()

        #plot in each iteration for each cut
        ax[j] = dic_cumsum['TestGroup_Cut' + str(p)].plot(figsize = (15, 5))
        ax[j].axhline(y = 0.95, linewidth = 1, color = 'r')
        ax[j].grid(True)
        ax[j].legend(loc = 'right')
        ax[j].set_title(data_type + '_' + denominator + '_' + cut + '_' + str(p))
        ax[j].set_xlabel('date')
        ax[j].set_ylabel('sig_level')
        y_major_locator = MultipleLocator(0.05) #set scale interval as 0.05 for the plot
        ax[j].yaxis.set_major_locator(y_major_locator)
        
        j = j + 1

    return 