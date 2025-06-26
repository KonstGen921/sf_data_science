## game guess the number ##

import numpy as np

number = np.random.randint(1,101)

count=0
while True:
    count+=1
    predict_number = int(input())
    
    if predict_number > number:
        print('number should be less')
        
    elif predict_number < number:
        print('number should be more')
        
    else:
        print(f'you are wright {number}, {count}')
        
        break