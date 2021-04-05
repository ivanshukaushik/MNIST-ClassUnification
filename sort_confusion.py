import confusion_matrix as cm
import numpy as np
import pandas as pd

def sort_confusion_actual(df1):
    actual_cf = cm.greatest_confusion_actual(df1)
    #predicted_cf = cm.greatest_confusion_predicted(cm.create_confusion_matrix()[0])

    numbers = df1.columns
    zip_acf = dict(zip(numbers, actual_cf))
    #zip_pcf = dict(zip(numbers, predicted_cf))
    

    return (zip_acf) #, (zip_pcf)


def find_most_confusing_actual(dic, df):
    
    max = 0
    max_num = 0
    for key, val in dic.items():
        if val > max_num:
            max_num = val
            max = key
    
    print(max, 'maximum')
    concerned_row = np.array(df.iloc[max])
    numbers = df.columns

    zip_conc = dict(zip(numbers, concerned_row))
    keys = list(zip_conc.keys())
    keys.sort(key=lambda x: zip_conc[x],reverse=True)
    max_conf = keys[1:2] # just the keys
  
    return max, max_conf[0]


def remove_most_confusing_actual(max, max_conf, y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    y_pred[y_pred == max_conf] = max
    y_true[y_true == max_conf] = max
    #print(cm.conf_matrix(y_pred, y_true))
    new_df = cm.conf_matrix(y_pred, y_true)
    return new_df, y_pred, y_true



dic = sort_confusion_actual(cm.create_confusion_matrix()[0])
_, _, _, y_test = cm.load_dataset()
df, y_pred = cm.create_confusion_matrix()
print(df)
m, mc = find_most_confusing_actual(dic, cm.create_confusion_matrix()[0])
new, new_y_pred, new_y_true = remove_most_confusing_actual(m, mc, y_pred, y_test)

# dic2 = sort_confusion_actual(new)
print(new)
# m, mc = find_most_confusing_actual(dic2, new)
# new2, new2_y_pred, new2_y_true = remove_most_confusing_actual(m, mc, new_y_pred, new_y_true)

# dic3 = sort_confusion_actual(new2)
# print(new2)
# m, mc = find_most_confusing_actual(dic3, new2)
# new3, new3_y_pred, new3_y_true = remove_most_confusing_actual(m, mc, new2_y_pred, new2_y_true)

