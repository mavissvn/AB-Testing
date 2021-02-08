#get all packages we need
import pandas as pd
import math
import scipy.stats
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

#read the data 
#download data here: https://drive.google.com/drive/folders/15fAv8MUw1J0k7FmRPNNK5NuseqGgSLhu?usp=sharing
df_test_overall = pd.read_csv('./AB Visit Data Test1_2.csv')
df_cateforymap = pd.read_csv('./MTA AB Category Mapping.csv')


#pull the data we need (test2)
df_test = df_test_overall[df_test_overall['testid'] == 2]
df_test = df_test.merge(df_cateforymap, how = 'left', on = 'ChannelID')
df_test['Channel Name'] = df_test['Channel Name'].fillna('Undefined')

#check on data to see if there are missing data on some specific days
df_test.drop_duplicates(['Date', 'SessionID']).groupby('Date')['SessionID'].count()

#select the date range with stable daily traffic according to the previous step
df_test = df_test[pd.to_datetime(df_test['Date']) > pd.to_datetime('2019-06-15')]
df_test = df_test[pd.to_datetime(df_test['Date']) < pd.to_datetime('2019-07-12')] 

#define z test calculator
def z_test_calculator (df, denominator, numerator):
    
    #get the data
    control_denominator = df.loc[1, denominator]
    control_numerator = df.loc[1, numerator]
    var_denominator = df.loc[0, denominator]
    var_numerator = df.loc[0, numerator]
    
    #get the rate
    control_rate = control_numerator / control_denominator
    var_rate = var_numerator / var_denominator
    
    #STD (actually it's square root of std^2/n, we do this because we can directly input it in the subsequent calculation
    control_std = math.sqrt(abs(control_rate * (1 - control_rate) / control_denominator))
    var_std = math.sqrt(abs(var_rate * (1 - var_rate) / var_denominator))
    
    #z score
    z_score = (control_rate - var_rate) / math.sqrt(pow(control_std, 2) + pow(var_std, 2))
    
    #p value
    p_value = scipy.stats.norm.sf(abs(z_score))
    
    #lift
    perc_lift = (var_rate - control_rate) / control_rate
    abs_lift = var_rate - control_rate
    
    return (p_value, perc_lift, abs_lift)


# # A/B Test (Binomial)
#define ab test calculator by using some loops to save analysis execution time
def ab_test_calculator (df, data_type): #df = df_test, data_type = ['SessionID', 'CusID']

    #integrate all necesary input data into df_result by using loop
    metrics = ['SawProduct', 'Bounced', 'AddedToCart', 'ReachedCheckout', 'Converted']

    df_result = df.drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[[data_type]].count()

    for metric in metrics:
        a = df[df[metric] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[data_type].count()
        a.name = metric
        df_result = df_result.join(a)

    #input specific KPIs to get coresponding outputs by using z_test_calculator & fill in all outputs in df_final
    KPIs = [(data_type, 'SawProduct'), 
            (data_type, 'Bounced'),
            (data_type, 'AddedToCart'),
            (data_type, 'ReachedCheckout'),
            (data_type, 'Converted'),
            ('SawProduct', 'AddedToCart'),
            ('AddedToCart', 'ReachedCheckout'),
            ('ReachedCheckout', 'Converted')]

    dic_final = {}

    df_final = pd.DataFrame()

    j = 0
    for i in KPIs:
        result = z_test_calculator (df_result, i[0], i[1])
        df_final.loc[j, 'demoninator'] = i[0]
        df_final.loc[j, 'numerator'] = i[1]
        df_final.loc[j, 'p_value'] = result[0]
        df_final.loc[j, 'perc_lift'] = result[1]
        df_final.loc[j, 'abs_lift'] = result[2]
        j = j + 1

    dic_final['TestGroup'] = df_final
    
    return df_final

#define reference line tool to realize automation
def reference_line(df, denominator, numerator, data_type): #df = df_test, denominator/numerator = ['SessionID'~'Converted'], data_type = ['SessionID', 'CusID']
    
    #get daily data for denominator
    if denominator == data_type:
        denominator_dailydata = df.drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()
    
    elif denominator != data_type:    
        denominator_dailydata = df[df[denominator] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()    
    
    #get daily data for numerator
    numerator_dailydata = df[df[numerator] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()

    #name them
    denominator_dailydata.name = denominator
    numerator_dailydata.name = numerator

    #reset index (the original index is 'Date')
    denominator_aggdailydata = denominator_dailydata.reset_index()
    numerator_aggdailydata = numerator_dailydata.reset_index()

    #replace denominator/numearator_aggdailydata's original denominator/ numerator column data with cumsum data
    denominator_aggdailydata[denominator] = denominator_aggdailydata.groupby('ControlGroup')[denominator].cumsum()
    numerator_aggdailydata[numerator] = numerator_aggdailydata.groupby('ControlGroup')[numerator].cumsum()

    #calculate p value, perc lift & abs lift for every day by using loop and z test calculator, and then fill them in df_cumsum
    df_cumsum = pd.DataFrame()

    for date in df.drop_duplicates('Date')['Date'].tolist():
        df_result = denominator_aggdailydata[denominator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True)
        df_result = df_result.merge(numerator_aggdailydata[numerator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True), on = ['Date', 'ControlGroup'])

        sig_result = z_test_calculator(df_result, denominator, numerator)
        df_cumsum.loc[date, 'sig_level'] = 1 - sig_result[0]
        df_cumsum.loc[date, 'perc_lift'] = sig_result[1]
        df_cumsum.loc[date, 'abs_lift'] =  sig_result[2]

    #tidy up the order
    df_cumsum.sort_index()

    #plot
    ax = df_cumsum.sort_index()['sig_level'].plot(figsize = (15, 5))
    ax.axhline(y = 0.95, linewidth = 1, color = 'r')
    ax.title
    ax.grid(True)
    ax.legend(loc = 'right')
    ax.set_title(data_type + '_' + denominator + '_' + numerator)
    ax.set_xlabel('date')
    ax.set_ylabel('sig_level')

    return ax


# # A/B Test (Binomial) w/ cut
#from here we define a series of a/b test w/ cut automated functions in similar logic as the above
#define ab test calculator applied to cut
def ab_test_calculator_cut (df, data_type, cut): #df = df_test, data_type = ['SessionID', 'CusID'], cut = ['CategoryID', 'VisitorType', 'Channel Name']
    
    dic_final = {}
    df_cut = df.copy()
    
    #use loop to get results cut by cut
    for p in set(df_cut[cut]):
        
        #integrate all necesary input data into df_result by using loop
        df = df_cut[df_cut[cut] == p]

        metrics = ['SawProduct', 'Bounced', 'AddedToCart', 'ReachedCheckout', 'Converted']

        df_result = df.drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[[data_type]].count()

        for metric in metrics:
            a = df[df[metric] == 1].drop_duplicates([data_type, 'ControlGroup']).groupby('ControlGroup')[data_type].count()
            a.name = metric
            df_result = df_result.join(a)


        #input specific KPIs to get coresponding outputs by using z_test_calculator & fill in all outputs in df_final
        KPIs = [(data_type, 'SawProduct'), 
                (data_type, 'Bounced'),
                (data_type, 'AddedToCart'),
                (data_type, 'ReachedCheckout'),
                (data_type, 'Converted'),
                ('SawProduct', 'Bounced'),
                ('SawProduct', 'AddedToCart'),
                ('AddedToCart', 'ReachedCheckout'),
                ('ReachedCheckout', 'Converted')]

        df_final = pd.DataFrame()
        j = 0
        
        for i in KPIs:
            result = z_test_calculator (df_result, i[0], i[1])
            df_final.loc[j, 'demoninator'] = i[0]
            df_final.loc[j, 'numerator'] = i[1]
            df_final.loc[j, 'p_value'] = result[0]
            df_final.loc[j, 'perc_lift'] = result[1]
            df_final.loc[j, 'abs_lift'] = result[2]
            j = j + 1

        dic_final[cut + '_' + str(p)] = df_final
    
    return dic_final


#define reference line tool applied to cut
def reference_line_cut(df, denominator, numerator, data_type, cut): #df = df_test, denominator/numerator = ['SawProduct'~'Converted'], data_type = ['SessionID', 'CusID'], cut = ['CategoryID', 'VisitorType', 'Channel Name']
    
    dic_cumsum = {}
    
    j = 0
    
    fig, ax = plt.subplots(len(set(df[cut])))
    
    #use loop to get results cut by cut
    for p in set(df[cut]):
        
        #get daily data for denominator
        if denominator == data_type:
            denominator_dailydata = df[df[cut] == p].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()
        
        elif denominator != data_type:
            denominator_dailydata = df[(df[cut] == p) & (df[denominator] == 1)].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()
    	
    	#get daily data for numerator
        numerator_dailydata = df[(df[cut] == p) & (df[numerator] == 1)].drop_duplicates([data_type, 'ControlGroup']).groupby(['Date', 'ControlGroup'])[data_type].count().unstack(fill_value = 0).stack()

        #name them
        denominator_dailydata.name = denominator
        numerator_dailydata.name = numerator

        #reset index (the original index is 'Date')
        denominator_aggdailydata = denominator_dailydata.reset_index()
        numerator_aggdailydata = numerator_dailydata.reset_index()

        #replace denominator/numearator_aggdailydata's original denominator/ numerator column data with cumsum data
        denominator_aggdailydata[denominator] = denominator_aggdailydata.groupby('ControlGroup')[denominator].cumsum()
        numerator_aggdailydata[numerator] = numerator_aggdailydata.groupby('ControlGroup')[numerator].cumsum()

        #calculate p value, perc lift & abs lift for every day by using loop and z test calculator, and then fill them in df_cumsum
        df_cumsum = pd.DataFrame()

        for date in df.drop_duplicates('Date')['Date'].tolist():
            df_result = denominator_aggdailydata[denominator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True)
            df_result = df_result.merge(numerator_aggdailydata[numerator_aggdailydata['Date'] == date].set_index('ControlGroup', drop = True), on = ['Date', 'ControlGroup'])

            sig_result = z_test_calculator(df_result, denominator, numerator)
            df_cumsum.loc[date, 'sig_level'] = 1 - sig_result[0]
            df_cumsum.loc[date, 'perc_lift'] = sig_result[1]
            df_cumsum.loc[date, 'abs_lift'] =  sig_result[2]

        #tidy up the order & update df_cumsum data in dic_cumsum in each iteration
        dic_cumsum['TestGroup_Cut' + str(p)] = df_cumsum.sort_index()

        #plot in each iteration for each cut
        ax[j] = dic_cumsum['TestGroup_Cut' + str(p)].plot(figsize = (15, 5))
        ax[j].axhline(y = 0.95, linewidth = 1, color = 'r')
        ax[j].grid(True)
        ax[j].legend(loc = 'right')
        ax[j].set_title(cut+ '_' + str(p))
        ax[j].set_xlabel('date')
        ax[j].set_ylabel('sig_level')

        j = j + 1
        
    return ax
