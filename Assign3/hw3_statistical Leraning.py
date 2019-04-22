import pandas as pd
import numpy as np
import pickle
from sklearn import metrics, linear_model
from sklearn.model_selection import GridSearchCV, train_test_split

with open('/Users/ryanhuang/Desktop/107-2/statistical learning/Assignments/Assign3/phone_train.pickle', 'rb') as fh1:
    traindata = pickle.load(fh1)

with open('/Users/ryanhuang/Desktop/107-2/statistical learning/Assignments/Assign3/phone_test1.pickle', 'rb') as fh2:
    test = pickle.load(fh2)

x_train = traindata.iloc[:,0:3]
y_train = traindata.iloc[:,3]

x_test = test.iloc[:,0:3]
y_test = test.iloc[:,3]


class mypgc:
    def fit(self, X_train, Y_train):
        def model_info(x_train,y_train, ylab):
            info_collection = dict()
            for lab in ylab:
                info = dict()
                data = x_train.loc[y_train == lab,]
                info['mu'] = data.mean(0)
                info['cov'] = data.cov(0)
                info['prec'] = np.linalg.inv(data.cov(0))
                info['detcov'] = np.linalg.det(data.cov(0))
                info['n'] = data.shape[0]
                info['prior'] = data.shape[0] / x_train.shape[0]
                info_collection[lab] = info
            return info_collection
        self.xtrain = X_train
        self.ytrain = Y_train
        self.ylab = self.ytrain.unique()
        self.model = model_info(self.xtrain,self.ytrain, self.ylab)

    def predict(self, x_test):
        prediction = []
        for i in range(x_test.shape[0]):
            p_list = []
            for lab in self.ylab:
                p_list.append(((2 * np.pi) ** (-3 / 2)) \
                              * (self.model[lab]["detcov"] ** (-1 / 2)) \
                              * np.exp(
                                        (-0.5) * (x_test.iloc[i,:] - self.model[lab]["mu"]).T \
                                        .dot(self.model[lab]["prec"]) \
                                        .dot(x_test.iloc[i,:] - self.model[lab]["mu"])) \
                              * self.model[lab]["prior"])
            prediction.append(self.ylab[np.asarray(p_list).argmax()])
        return np.asarray(prediction)

pgc1 = mypgc()
pgc1.fit(x_train, y_train)
pgc1.model['Phoneonback']['cov']
pred = pgc1.predict(x_test.iloc[0:20,])
print(pgc1.model)

def accuracy(pred, actual):
    acc = 0
    for i,j in zip(pred, actual):
        if i == j:
            acc += 1
    return acc / len(pred)

accuracy(pred, y_test.iloc[0:20,])


# Question 2

# Q2-1 Data Cleaning

# read the data, N/A values are '?'
header_list = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
               'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country', 'attribute']
q2_train_path = '/Users/ryanhuang/Desktop/107-2/statistical learning/Assignments/Assign3/adult.data.txt'
q2_train_raw = pd.read_csv(q2_train_path, header = None, names = header_list, na_values=['?'], skipinitialspace=True)
q2_test_path = '/Users/ryanhuang/Desktop/107-2/statistical learning/Assignments/Assign3/adult.test.txt'
q2_test_raw = pd.read_csv(q2_test_path, header = None, names = header_list, na_values=['?'], skipinitialspace=True)
q2_train_raw.dropna(axis=0, inplace=True)
q2_test_raw.dropna(axis=0, inplace=True)
q2_test_raw['age'] = q2_test_raw['age'].astype(int)
# Clean the category variables with 1 hot encoding
class_var_list = ['workclass', 'education', 'marital-status','occupation','relationship','race','native-country','sex']
class_others = []
small_class_col = []
for col in class_var_list:
    other_value = q2_train_raw[col].value_counts()\
                         .index\
                         [q2_train_raw[col].value_counts()<10]
    if len(other_value) > 0:
        class_others.append(other_value[:len(other_value)])
        small_class_col.append(col)
for col in small_class_col:
    q2_train_raw.loc[:,col] = q2_train_raw.loc[:,col].apply(lambda x: 'others' if x in class_others else x)
    q2_test_raw.loc[:,col] = q2_test_raw.loc[:,col].apply(lambda x: 'others' if x in class_others else x)


# 1-hot encoding for class_vars
# The conti. vars are from column_1 to column_7, column_8 is the attribute
q2_train = pd.get_dummies(q2_train_raw)
q2_test  = pd.get_dummies(q2_test_raw)

# check different columns in test and training
train_col = q2_train.columns.tolist()
test_col = q2_test.columns.tolist()
print(list(set(train_col).difference(set(test_col)))) # in train not in test
print(list(set(test_col).difference(set(train_col)))) # in test not in train

# add native-country_others to testing data
q2_test['native-country_others'] = 0


# rename columns
q2_test = q2_test.rename({'attribute_<=50K.':'attribute_<=50K','attribute_>50K.':'attribute_>50K'}, axis='columns')

# Rearrange the col order
q2_test = q2_test[q2_train.columns]

# seperate train_x, train_y, test_x, test_y
x_train, y_train = np.asarray(q2_train.iloc[:,:-2]), np.asarray(q2_train.iloc[:,-1])
x_test, y_test   = np.asarray(q2_test.iloc[:,:-2]), np.asarray(q2_test.iloc[:,-1])

# x_train
q2_train.iloc[:,:-2].describe()
# y_train
q2_train.iloc[:,-1].describe()
# x_test
q2_test.iloc[:,:-2].describe()
# y_test
q2_test.iloc[:,-1].describe()

# Q_2.2
# Gradient = Λw + ∑(yn - tn)xn
# Hession = Λ + x^T * R * x, where R is NxN diagonal matrix with the diagonal is yn * ( 1-yn )

# Q_2.3
class mylogistic_l2:
    def __init__(self, reg_vec, max_iter, tol, add_intercept = True):
        self.reg_vec = reg_vec
        self.max_iter = max_iter
        self.tol = tol
        self.add_intercept = add_intercept

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x_train, y_train):
        if self.add_intercept:
            self.x_train = np.c_[ x_train, np.ones(x_train.shape[0]) ]
        else:
            self.x_train = x_train
        self.y_train = y_train

        # initial w vector given by ridge regression
        w = np.linalg.inv( self.x_train.T.dot(self.x_train) +
                                (np.diag(self.reg_vec).mean()
                                    * np.identity(self.x_train.shape[1]))) \
                .dot(self.x_train.T) \
                .dot(self.y_train)

        # IRLS Loop to find optimal w
        error = []
        for i in range(self.max_iter):
            y = self.__sigmoid(np.dot(self.x_train, w)) # y = n(data size) * 1, each element is in [0,1]
            # y = 1 / (1 + np.exp((-self.w).T.dot(self.x_train.T)))
            t = self.y_train  # t = n(data size) * 1
            gradient = self.reg_vec.dot(w) + self.x_train.T.dot(y - t)  # gradient = m(feature size) * 1
            R = np.diagflat(y * (1 - y))
            hession = self.reg_vec + self.x_train.T.dot(R).dot(self.x_train)
            w = w - np.linalg.inv(hession).dot(gradient)
            new_error = 0.5 * w.T.dot(self.reg_vec).dot(w) - (t * np.log(y) + (1 - t * np.log((1 - y)))).sum()

            if i > 0 and abs(new_error - error[-1]) <= self.tol:
                break
            else:
                error.append(new_error)

        # w is determined after the IRLS converged
        self.w = w
        self.y = y

    def predict(self, x_test):
        if self.add_intercept:
            self.x_test = np.c_[x_test, np.ones(x_test.shape[0])]
        else:
            self.x_test = x_test
        pred = self.__sigmoid(np.dot(self.x_test, self.w))
        return pred.round()

# case_1: lambda = 1 for all coefficients
lambda_vec_1 = np.diagflat([1] * (x_train.shape[1]+1))
logic1 = mylogistic_l2(reg_vec = lambda_vec_1, max_iter = 1000, tol = 1e-5, add_intercept = True)
logic1.fit(x_train, y_train)
ypred1 = logic1.predict(x_test)
print('accuracy is', metrics.accuracy_score(y_test, ypred1))
print('coefficients w are', logic1.w)



w = logic1.w
reg_vec = logic1.reg_vec
t = logic1.y_train
y = logic1.y
obj_value_new = np.sum(np.square(w) * reg_vec) * 0.5 - np.sum(t * np.log(y) + (1-t) * np.log(1-y))

new_error = 0.5 * w.T.dot(reg_vec).dot(w) - ((t * np.log(y) + (1 - t) * np.log(1 - y))).sum()
obj_value_new
new_error



#case_2: lambda = 1 for all but the intercept, no regularization for intercept term.
lambda_vec_2 = np.diagflat([1] * x_train.shape[1] + [0])
logic2 = mylogistic_l2(reg_vec = lambda_vec_2, max_iter = 1000, tol = 1e-5, add_intercept = True)
logic2.fit(x_train, y_train)
ypred2 = logic2.predict(x_test)
print('accuracy is', metrics.accuracy_score(y_test, ypred2))
print('coefficients w are', logic2.w)

#case_3: lambda = 1 for numerical-valued features, lambda = 0.5 for binary-valued features, no regularization for incercept
lambda_vec_3 = np.diagflat([1] * 7 + [0.5] *(x_train.shape[1]-7) + [0])
logic3 = mylogistic_l2(reg_vec = lambda_vec_3, max_iter = 1000, tol = 1e-5, add_intercept = True)
logic3.fit(x_train, y_train)
ypred3 = logic3.predict(x_test)
metrics.accuracy_score(y_test, ypred3)
print('accuracy is', metrics.accuracy_score(y_test, ypred3))
print('coefficients w are', logic3.w)

# Q_2.4
def lambda_vec(a1, a2):
    return np.diagflat([a1] * 7 + [a2] * 97 + [0])

x_subtrain, x_tuning, y_subtrain, y_tuning = train_test_split(x_train,y_train, train_size=0.9)
lambda_list = np.logspace(-2,2,10)
# stage_1
stage_1 = []
for i in lambda_list:
    q2_4_logic = mylogistic_l2(max_iter = 1000, tol = 1e-5, add_intercept = True, reg_vec=lambda_vec(i,i))
    q2_4_logic.fit(x_subtrain, y_subtrain)
    ypred = q2_4_logic.predict(x_tuning)
    stage_1.append(metrics.accuracy_score(y_tuning, ypred))

print('a1, a2 are', lambda_list[stage_1.index(max(stage_1))])

# stage 2: fixed a1

stage_2 = []
for i in lambda_list:
    q2_4_logic = mylogistic_l2(max_iter = 1000, tol = 1e-5, add_intercept = True, reg_vec=lambda_vec(lambda_list[stage_1.index(max(stage_1))],i))
    q2_4_logic.fit(x_subtrain, y_subtrain)
    ypred = q2_4_logic.predict(x_tuning)
    stage_2.append(metrics.accuracy_score(y_tuning, ypred))
print('new a2 is', lambda_list[stage_2.index(max(stage_2))])

# stage 3: fixed a2
stage_3 = []
for i in lambda_list:
    q2_4_logic = mylogistic_l2(max_iter = 1000, tol = 1e-5, add_intercept = True, reg_vec=lambda_vec(i,lambda_list[stage_1.index(max(stage_2))]))
    q2_4_logic.fit(x_subtrain, y_subtrain)
    ypred = q2_4_logic.predict(x_tuning)
    stage_3.append(metrics.accuracy_score(y_tuning, ypred))
print('new a1 is', lambda_list[stage_3.index(max(stage_3))])

# Final summary
q2_4_logic_final = mylogistic_l2(max_iter = 1000, tol = 1e-5, add_intercept = True, reg_vec=lambda_vec(lambda_list[stage_1.index(max(stage_3))],lambda_list[stage_1.index(max(stage_2))]))
q2_4_logic_final.fit(x_train, y_train)
q2_5_logic_final_pred = q2_4_logic_final.predict(x_test)
print('The best pair of (a1, a2) is (',lambda_list[stage_1.index(max(stage_3))],',',lambda_list[stage_1.index(max(stage_2))],')')
print('The resulting accuracy is', metrics.accuracy_score(y_test, q2_5_logic_final_pred))

# Q_2.5
grid={"C":np.logspace(-2,2,10),"penalty":["l2"]}
q2_5_logic = linear_model.LogisticRegression(solver='lbfgs',max_iter=1000)
logreg_cv=GridSearchCV(q2_5_logic,grid)
logreg_cv.fit(x_subtrain,y_subtrain)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
q2_5_logic_final = linear_model.LogisticRegression(C=logreg_cv.best_params_['C'], penalty='l2')
q2_5_logic_final.fit(x_train, y_train)
pred = q2_5_logic_final.predict(x_test)
metrics.accuracy_score(y_test, pred)

# Compared with the standard model, our own model performances better with 5% higher accuracy score
# However, when it comes to efficiency, the standard model is so much faster
# Comparing the coefficients:
print(q2_5_logic_final.coef_)
print(q2_4_logic_final.w)

accu = []
for c in np.logspace(-2,2,10):
    q2_5_logic = linear_model.LogisticRegression(solver='lbfgs', C=c, max_iter=1000)
    q2_5_logic.fit(x_subtrain,y_subtrain)
    pred = q2_5_logic.predict(x_tuning)
    accu.append(metrics.accuracy_score(y_tuning, pred))

accu.index(max(accu))
np.logspace(-2,2,10)[5]
