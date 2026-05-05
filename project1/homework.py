names = ['chlorhexidine', 'cyntomycin', 'afobazol'] 
counts = [15, 18, 7]
import pandas as pd
def create_medications(names, counts):
    medications= pd.Series(data = counts, index=names, name='medications')
    return medications
med_series = create_medications(names, counts)
def get_percent(med_series):
    total = med_series.sum()
    percent_series = (med_series/total)
    return(percent_series)
percent_series= get_percent(med_series)
print(med_series, percent_series, sep='\n')