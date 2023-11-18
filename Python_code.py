import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
import time
import csv



class DecisionTree:
    def __init__(self, maxDepth=None, minSize=2, 
                 missingValues='median', criterion='gini'):
        self.maxDepth = maxDepth
        self.minSize = minSize
        self.tree = None
        self.missingValues = missingValues
        self.MedianValue = None
        self.criterion = criterion

    def gini(self, groups, classes):
#function to calculate the Gini index 
        t_instances = float(sum([len(g) for g in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / t_instances)
        return gini

    def entropy(self, groups, classes):
#function to calculate entropy        
        t_instances = float(sum([len(group) for group in groups]))
        entropy = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                if p != 0:
                    score += -p * np.log2(p)
            entropy += score * (size / t_instances)
        return entropy

    def handle_missing_values(self, X_train):
#function to handle missing values:    
        self.Median_value = np.nanmedian(X_train, axis=0)
        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            mask = np.isnan(col)
            col[mask] = self.Median_value[i]
            X_train[:, i] = col
        return X_train

    def test_split(self, index, value, dataset):
#function to split the dataset based on an attribute value
        left, right = list(), list()
        for row in dataset:
            if np.isnan(row[index]):
                continue
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def get_split(self, dataset):
#function to Select the best split point.
        class_values = list(set(row[-1] for row in dataset))
        b_index= None
        b_value= None 
        b_score = None
        b_groups = None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                if self.criterion == 'entropy':
                    score = self.entropy(groups, class_values)
                elif self.criterion == 'gini':
                    score = self.gini(groups, class_values)
                if b_score is None or score < b_score:
                    b_index,b_value, b_score, b_groups = index, row[index], score, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(self, group):
#function to get the most common output value
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, maxDepth, minSize, depth):
#function to create child splits
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if maxDepth == None:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        elif depth >= maxDepth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # check for min size
        if len(left) <= minSize:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], maxDepth, minSize, depth+1)
        if len(right) <= minSize:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], maxDepth, minSize, depth+1)

            
    def fit(self, data, y):
#function to train the
        X_train = self.handle_missing_values(data)
        self.tree = self.get_split(np.column_stack((X_train, y)))
        self.split(self.tree, self.maxDepth, self.minSize, 1)
        
    def predict(self,X):
        return [self.predict_all(i) for i in X]   

    def predict_all(self, x):
#function to make predictions according to the training
        return self.make_prediction(self.tree, x)
    
    def make_prediction(self, node, x):
            if np.isnan(x[node['index']]):
                return self.to_terminal(node['groups'][0] + node['groups'][1])
            elif x[node['index']] < node['value']:
                if isinstance(node['left'], dict):
                    return self.make_prediction(node['left'], x)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return self.make_prediction(node['right'], x)
                else:
                    return node['right']

#=====================================================================================================================================================================================
#first dataframe "Haberman's Survival" from UCI machine learning repository website.           

df_haberman = pd.read_csv("haberman.data")

labels_haberman = df_haberman.iloc[:,-1:].to_numpy()
df_haberman = df_haberman.iloc[:, :-1].to_numpy()

X_train_df1, X_test_df1, y_train_df1, y_test_df1 = train_test_split(df_haberman, labels_haberman, test_size=0.2, random_state=55)


start = time.time()

dt = DecisionTree()
dt.fit(X_train_df1,y_train_df1)

y_pred_dt_haberman = np.array(dt.predict(X_test_df1))

end = time.time()

time_taken_dt_1 = end-start

print("accuracy haberman dataset dt: ", accuracy_score(y_test_df1, y_pred_dt_haberman))
print("time taken haberman dataset dt: ",time_taken_dt_1)



dtc = DecisionTreeClassifier()

dtc.fit(X_train_df1,y_train_df1)

y_pred_skdt_haberman = np.array(dtc.predict(X_test_df1))

end = time.time()
time_taken_skdt_1 = end-start
print("accuracy haberman dataset skdt: ", accuracy_score(y_test_df1, y_pred_skdt_haberman))
print("time taken haberman dataset skdt: ", time_taken_skdt_1)

#=====================================================================================================================================================================================
#second data frame "Mammographic Mass" from UCI machine learning repository website with missing values.           

df_mammographic= pd.read_csv("mammographic_masses.data")

df_mammographic.replace('?', np.nan,inplace = True)

labels_mammohraphic_dt = df_mammographic.iloc[:,-1:].to_numpy()

df_mammographic_dt = df_mammographic.iloc[:, :-1].to_numpy()

df_mammographic_dt = df_mammographic_dt.astype(float)
labels_mammohraphic_dt = labels_mammohraphic_dt.astype(float)


X_train_df2_dt, X_test_df2_dt, y_train_df2_dt, y_test_df2_dt = train_test_split(df_mammographic_dt, labels_mammohraphic_dt, test_size=0.2, random_state=55)


start = time.time()

dt = DecisionTree(criterion='entropy')
dt.fit(X_train_df2_dt,y_train_df2_dt)

y_pred_dt_mammographic = np.array(dt.predict(X_test_df2_dt))

end = time.time()

time_taken_dt_1 = end-start

print("accuracy mammographic dataframe dt: ", accuracy_score(y_test_df2_dt, y_pred_dt_mammographic))
print("time taken mammographic dataframe dt: ",time_taken_dt_1)


#As decision from scikit-learn tree does not support missing values we need pre processing to implement scikit learn.

df_mammographic_skdt = df_mammographic.fillna(df_mammographic.median())
df_mammographic_skdt = df_mammographic_skdt.astype(float)  
labels_mammohraphic_skdt = df_mammographic_skdt.iloc[:,-1:].to_numpy()

df_mammographic_skdt = df_mammographic_skdt.iloc[:, :-1].to_numpy()

dtc = DecisionTreeClassifier(criterion="entropy")

X_train_df2_skdt, X_test_df2_skdt, y_train_df2_skdt, y_test_df2_skdt = train_test_split(df_mammographic_skdt, labels_mammohraphic_skdt, test_size=0.2, random_state=55)

start = time.time()


dtc.fit(X_train_df2_skdt,y_train_df2_skdt)

y_pred_skdt_mammographic = np.array(dtc.predict(X_test_df2_skdt))

end = time.time()
time_taken_skdt_1 = end-start
print("accuracy mammographic dataframe skdt: ", accuracy_score(y_test_df2_skdt, y_pred_skdt_mammographic))
print("time taken mammographic dataframe skdt: ", time_taken_skdt_1)



#=====================================================================================================================================================================================


file = open('Predictions_Haberman.csv', 'w', newline='')

writer = csv.writer(file)

writer.writerow([x+1 for x in range(y_test_df1.shape[0])])
writer.writerow(y_test_df1.ravel())
writer.writerow(y_pred_dt_haberman.ravel())
writer.writerow(y_pred_skdt_haberman.ravel())


file.close()



file = open('Results_Haberman.csv', 'w', newline='')
header = ["accuracy_dt", "f1_dt", "recall_dt","precision_dt","time_taken_dt",
          "accuracy_skdt","f1_skdt","recall_skdt","precision_skdt","time_taken_skdt","max_depth"]
writer = csv.writer(file)
writer.writerow(header)


for i in range(1,15):

        dt = DecisionTree(max_depth=i)
        skdt = DecisionTreeClassifier(max_depth=i)

        start_dt = time.time()

        dt.fit(X_train_df1, y_train_df1)
        y_pred_dt_haberman = dt.predict(X_test_df1)

        end_dt = time.time()

        time_taken_dt_1 = end_dt - start_dt

        start_skdt = time.time()

        skdt.fit(X_train_df1, y_train_df1)
        y_pred_skdt_haberman = skdt.predict(X_test_df1)

        end_skdt = time.time()

        time_taken_skdt_1 =  end_skdt - start_skdt 

        data = [accuracy_score(y_test_df1, y_pred_dt_haberman), f1_score(y_test_df1, y_pred_dt_haberman), recall_score(y_test_df1, y_pred_dt_haberman),precision_score(y_test_df1, y_pred_dt_haberman),time_taken_dt_1,
                accuracy_score(y_test_df1,y_pred_skdt_haberman),f1_score(y_test_df1,y_pred_skdt_haberman),recall_score(y_test_df1,y_pred_skdt_haberman),precision_score(y_test_df1, y_pred_skdt_haberman),time_taken_skdt_1,i]
        writer.writerow(data)

file.close()
#=====================================================================================================================================================================================
#df mammographic:




file = open('Predictions_mammographic.csv', 'w', newline='')

writer = csv.writer(file)

writer.writerow([x+1 for x in range(y_test_df2_skdt.shape[0])])
writer.writerow(y_test_df2_skdt.ravel())
writer.writerow(y_pred_dt_mammographic.ravel())
writer.writerow(y_pred_skdt_mammographic.ravel())


file.close()



file = open('Results_mammographic.csv', 'w', newline='')
header = ["accuracy_dt", "f1_dt", "recall_dt","precision_dt","time_taken_dt",
          "accuracy_skdt","f1_skdt","recall_skdt","precision_skdt","time_taken_skdt","min_sample_split"]
writer = csv.writer(file)
writer.writerow(header)


for i in range(2,15):

    dt = DecisionTree(criterion="entropy",min_size=i)
    skdt = DecisionTreeClassifier(criterion="entropy",min_samples_split=i)

    start_dt = time.time()

    dt.fit(X_train_df2_dt, y_train_df2_dt)
    y_pred_dt_2 = dt.predict(X_test_df2_dt)

    end_dt = time.time()

    time_taken_dt_2 = end_dt - start_dt

    start_skdt = time.time()

    skdt.fit(X_train_df2_skdt, y_train_df2_skdt)
    y_pred_skdt_2 = skdt.predict(X_test_df2_skdt)

    end_skdt = time.time()

    time_taken_skdt_2 =  end_skdt - start_skdt 

    data = [accuracy_score(y_test_df2_dt, y_pred_dt_2), f1_score(y_test_df2_dt, y_pred_dt_2), recall_score(y_test_df2_dt, y_pred_dt_2),precision_score(y_test_df2_dt, y_pred_dt_2),time_taken_dt_2,
            accuracy_score(y_test_df2_skdt,y_pred_skdt_2),f1_score(y_test_df2_skdt,y_pred_skdt_2),recall_score(y_test_df2_skdt,y_pred_skdt_2),precision_score(y_test_df2_skdt, y_pred_skdt_2),time_taken_skdt_2,i]
    writer.writerow(data)

file.close()

