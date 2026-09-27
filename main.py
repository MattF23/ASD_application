import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import pickle

def activityTwo(x_train, x_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #print("Original feature accuracy: ", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix svm: ")
    print(cm)

    #print("Classification report: ", classification_report(y_test, y_pred))


def activityThree(x_train, x_test, y_train, y_test, x, y):
    #10 fold cross validation and measure model accuracy (cross validation score)
    clf = svm.SVC()
    scores = cross_val_score(clf, x, y, cv = 10)

    print("original features cross validation", scores)

    #Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, n_jobs = -1)

    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_test)

    print("With hyperparameter tuning test train split: ", accuracy_score(y_test, y_pred))

    print("Hyperparameter tuning cross validation: ", cross_val_score(grid, x, y, cv = 10))

    #best params from first run {c: 10, gamma: 0.0001, kernel: rbf}


def activityFour(x_train, y_train, adl):
    x_new = SelectKBest(f_classif, k = 100)
    x_train_k_best = x_new.fit_transform(x_train, y_train)

    #7030
    clf = svm.SVC()
    x = adl[x_train.columns[x_new.get_support()]]
    y = adl['class']
   
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

    clf.fit(x_train_k_best, y_train)
    y_pred = clf.predict(x_test)

    print("Activity four 70-30 accuracy: ", accuracy_score(y_test, y_pred))
    
    #10 fold class validation
    print("cross val scores: ", cross_val_score(clf, x, y, cv = 10))

def activityFive(x, y, adl):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    pca = PCA(n_components = 10)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print("Acitvity five accuracy: ", accuracy_score(y_test, y_pred))

    #10 fold
    print("Cross val scores: ", cross_val_score(classifier, x, y, cv = 10))

def activitySeven(x_train, x_test, y_train, y_test, x, y):
    #ACTIVITY SEVEN
    #SGDclassifier
    clf = SGDClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix sgd: ")
    print(cm)
    
    #print("sgdClassifier accuracy: ", accuracy_score(y_test, y_pred))

    #print("SgdClassifier cross val: ", cross_val_score(clf, x, y, cv = 10))

    #Random Forrest
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix random forest: ")
    print(cm)

    #print("Random forest accuracy: ", accuracy_score(y_test, y_pred))

    #print("Random forest cross val: ", cross_val_score(clf, x, y, cv = 10))

    #MPLClassifier
    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix MPL: ")
    print(cm)

    #print("MLP accuracy: ", accuracy_score(y_test, y_pred))

    #print("MLP cross val: ", cross_val_score(clf, x, y, cv = 10))

def randomforest(x_train, y_train, x_test, y_test, x, y):
    #Oversampling minority class
    smote = SMOTE(random_state = 42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    
    clf = RandomForestClassifier()
    clf.fit(x_resampled, y_resampled)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix random forest oversampling minority class: ")
    print(cm)

    #Undersampling majority class
    tome = TomekLinks(sampling_strategy = 'majority')
    x_resampled, y_resampled = tome.fit_resample(x, y)

    clf = RandomForestClassifier()
    clf.fit(x_resampled, y_resampled)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix random forest undersampling majority class:")
    print(cm)

    #Adjusting class weights
    class_weights = {0: 0.3,
                     1: 0.5,
                     2: 0.2}
    
    svc = SVC(class_weight = class_weights)
    svc.fit(x, y)

    y_pred = svc.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix svc adjusting class weight:")
    print(cm)

def Studio4Activity5(x, y, x_test, y_test):
    #Oversampling minority class
    smote = SMOTE(random_state = 42)
    x_overSampled, y_overSampled = smote.fit_resample(x, y)
    
    #Undersampling majority class
    tome = TomekLinks(sampling_strategy = 'majority')
    x_underSampled, y_underSampled = tome.fit_resample(x, y)
    
    #test train split
    #oversampled
    clf = svm.SVC()
    clf.fit(x_overSampled, y_overSampled)

    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)
    print("test train split oversampled classification report:")
    print(classification_report(y_test, y_pred))

    print("test train split oversampled confusion matrix:")
    print(cm)

    #undersampled
    clf.fit(x_underSampled, y_underSampled)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("test train split undersampled classification report:")
    print(classification_report(y_test, y_pred))

    print("test train split undersampled confusion matrix:")
    print(cm)

    filename = 'myModel.pkl'

    pickle.dump(clf, open(filename, 'wb'))

    #Same but with a random forrest model
    smote = SMOTE(random_state = 42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    
    clf = RandomForestClassifier()
    clf.fit(x_resampled, y_resampled)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_pred, y_test)

    print("Confusion matrix random forest oversampling minority class: ")
    print(cm)

    filename = 'myModel2.pkl'
    pickle.dump(clf, open(filename, 'wb'))

def Studio4Activity6(filename):
    model = pickle.load(open(filename, 'rb'))

    df = pd.read_csv('ampc/w4.csv')

    features = ['acc_mean_x_right', 'acc_mean_y_right', 'acc_mean_z_right', 'acc_mean_xyz_right', 'acc_mean_xy_right', 'acc_mean_yz_right', 'acc_mean_zx_right', 'acc_mean_pitch_right', 'acc_mean_roll_right', 'acc_std_x_right', 'acc_std_y_right', 'acc_std_z_right', 'acc_std_xyz_right', 'acc_std_xy_right', 'acc_std_yz_right', 'acc_std_zx_right', 'acc_std_pitch_right', 'acc_std_roll_right', 'acc_auc_x_right', 'acc_auc_y_right', 'acc_auc_z_right', 'acc_auc_xyz_right', 'acc_auc_xy_right', 'acc_auc_yz_right', 'acc_auc_zx_right', 'acc_auc_pitch_right', 'acc_auc_roll_right', 'acc_max_x_right', 'acc_max_y_right', 'acc_max_z_right', 'acc_max_xyz_right', 'acc_max_xy_right', 'acc_max_yz_right', 'acc_max_zx_right', 'acc_max_pitch_right', 'acc_max_roll_right', 'acc_peak_x_right', 'acc_peak_y_right', 'acc_peak_z_right', 'acc_peak_xyz_right', 'acc_peak_xy_right', 'acc_peak_yz_right', 'acc_peak_zx_right', 'gyro_mean_x_right', 'gyro_mean_y_right', 'gyro_mean_z_right', 'gyro_mean_xyz_right', 'gyro_mean_xy_right', 'gyro_mean_yz_right', 'gyro_mean_zx_right', 'gyro_std_x_right', 'gyro_std_y_right', 'gyro_std_z_right', 'gyro_std_xyz_right', 'gyro_std_xy_right', 'gyro_std_yz_right', 'gyro_std_zx_right', 'gyro_auc_x_right', 'gyro_auc_y_right', 'gyro_auc_z_right', 'gyro_auc_xyz_right', 'gyro_auc_xy_right', 'gyro_auc_yz_right', 'gyro_auc_zx_right', 'gyro_max_x_right', 'gyro_max_y_right', 'gyro_max_z_right', 'gyro_max_xyz_right', 'gyro_max_xy_right', 'gyro_max_yz_right', 'gyro_max_zx_right', 'gyro_peak_x_right', 'gyro_peak_y_right', 'gyro_peak_z_right', 'gyro_peak_xyz_right', 'gyro_peak_xy_right', 'gyro_peak_yz_right', 'gyro_peak_zx_right', 'acc_mean_x_left', 'acc_mean_y_left', 'acc_mean_z_left', 'acc_mean_xyz_left', 'acc_mean_xy_left', 'acc_mean_yz_left', 'acc_mean_zx_left', 'acc_mean_pitch_left', 'acc_mean_roll_left', 'acc_std_x_left', 'acc_std_y_left', 'acc_std_z_left', 'acc_std_xyz_left', 'acc_std_xy_left', 'acc_std_yz_left', 'acc_std_zx_left', 'acc_std_pitch_left', 'acc_std_roll_left', 'acc_auc_x_left', 'acc_auc_y_left', 'acc_auc_z_left', 'acc_auc_xyz_left', 'acc_auc_xy_left', 'acc_auc_yz_left', 'acc_auc_zx_left', 'acc_auc_pitch_left', 'acc_auc_roll_left', 'acc_max_x_left', 'acc_max_y_left', 'acc_max_z_left', 'acc_max_xyz_left', 'acc_max_xy_left', 'acc_max_yz_left', 'acc_max_zx_left', 'acc_max_pitch_left', 'acc_max_roll_left', 'acc_peak_x_left', 'acc_peak_y_left', 'acc_peak_z_left', 'acc_peak_xyz_left', 'acc_peak_xy_left', 'acc_peak_yz_left', 'acc_peak_zx_left', 'gyro_mean_x_left', 'gyro_mean_y_left', 'gyro_mean_z_left', 'gyro_mean_xyz_left', 'gyro_mean_xy_left', 'gyro_mean_yz_left', 'gyro_mean_zx_left', 'gyro_std_x_left', 'gyro_std_y_left', 'gyro_std_z_left', 'gyro_std_xyz_left', 'gyro_std_xy_left', 'gyro_std_yz_left', 'gyro_std_zx_left', 'gyro_auc_x_left', 'gyro_auc_y_left', 'gyro_auc_z_left', 'gyro_auc_xyz_left', 'gyro_auc_xy_left', 'gyro_auc_yz_left', 'gyro_auc_zx_left', 'gyro_max_x_left', 'gyro_max_y_left', 'gyro_max_z_left', 'gyro_max_xyz_left', 'gyro_max_xy_left', 'gyro_max_yz_left', 'gyro_max_zx_left', 'gyro_peak_x_left', 'gyro_peak_y_left', 'gyro_peak_z_left', 'gyro_peak_xyz_left', 'gyro_peak_xy_left', 'gyro_peak_yz_left', 'gyro_peak_zx_left']

    x = df[features]
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

    #y_pred = model.predict(x_test)
    #print(confusion_matrix(y_pred, y_test))
    count = 0

    df['predictions'] = x.apply(lambda x: model.predict(x.values[None])[0], axis = 1)

    iterator = 0
    for i in df['predictions']:
        if i == y.iloc[i]:
            count += 1
        iterator += 1

    print("Number of correct classifications: ", count)

def main():
    df = pd.concat(map(pd.read_csv, ['ampc/w1.csv', 'ampc/w2.csv', 'ampc/w3.csv', 'ampc/w4.csv']), ignore_index = True)

    df.to_csv('combined_data.csv')

    features = ['acc_mean_x_right', 'acc_mean_y_right', 'acc_mean_z_right', 'acc_mean_xyz_right', 'acc_mean_xy_right', 'acc_mean_yz_right', 'acc_mean_zx_right', 'acc_mean_pitch_right', 'acc_mean_roll_right', 'acc_std_x_right', 'acc_std_y_right', 'acc_std_z_right', 'acc_std_xyz_right', 'acc_std_xy_right', 'acc_std_yz_right', 'acc_std_zx_right', 'acc_std_pitch_right', 'acc_std_roll_right', 'acc_auc_x_right', 'acc_auc_y_right', 'acc_auc_z_right', 'acc_auc_xyz_right', 'acc_auc_xy_right', 'acc_auc_yz_right', 'acc_auc_zx_right', 'acc_auc_pitch_right', 'acc_auc_roll_right', 'acc_max_x_right', 'acc_max_y_right', 'acc_max_z_right', 'acc_max_xyz_right', 'acc_max_xy_right', 'acc_max_yz_right', 'acc_max_zx_right', 'acc_max_pitch_right', 'acc_max_roll_right', 'acc_peak_x_right', 'acc_peak_y_right', 'acc_peak_z_right', 'acc_peak_xyz_right', 'acc_peak_xy_right', 'acc_peak_yz_right', 'acc_peak_zx_right', 'gyro_mean_x_right', 'gyro_mean_y_right', 'gyro_mean_z_right', 'gyro_mean_xyz_right', 'gyro_mean_xy_right', 'gyro_mean_yz_right', 'gyro_mean_zx_right', 'gyro_std_x_right', 'gyro_std_y_right', 'gyro_std_z_right', 'gyro_std_xyz_right', 'gyro_std_xy_right', 'gyro_std_yz_right', 'gyro_std_zx_right', 'gyro_auc_x_right', 'gyro_auc_y_right', 'gyro_auc_z_right', 'gyro_auc_xyz_right', 'gyro_auc_xy_right', 'gyro_auc_yz_right', 'gyro_auc_zx_right', 'gyro_max_x_right', 'gyro_max_y_right', 'gyro_max_z_right', 'gyro_max_xyz_right', 'gyro_max_xy_right', 'gyro_max_yz_right', 'gyro_max_zx_right', 'gyro_peak_x_right', 'gyro_peak_y_right', 'gyro_peak_z_right', 'gyro_peak_xyz_right', 'gyro_peak_xy_right', 'gyro_peak_yz_right', 'gyro_peak_zx_right', 'acc_mean_x_left', 'acc_mean_y_left', 'acc_mean_z_left', 'acc_mean_xyz_left', 'acc_mean_xy_left', 'acc_mean_yz_left', 'acc_mean_zx_left', 'acc_mean_pitch_left', 'acc_mean_roll_left', 'acc_std_x_left', 'acc_std_y_left', 'acc_std_z_left', 'acc_std_xyz_left', 'acc_std_xy_left', 'acc_std_yz_left', 'acc_std_zx_left', 'acc_std_pitch_left', 'acc_std_roll_left', 'acc_auc_x_left', 'acc_auc_y_left', 'acc_auc_z_left', 'acc_auc_xyz_left', 'acc_auc_xy_left', 'acc_auc_yz_left', 'acc_auc_zx_left', 'acc_auc_pitch_left', 'acc_auc_roll_left', 'acc_max_x_left', 'acc_max_y_left', 'acc_max_z_left', 'acc_max_xyz_left', 'acc_max_xy_left', 'acc_max_yz_left', 'acc_max_zx_left', 'acc_max_pitch_left', 'acc_max_roll_left', 'acc_peak_x_left', 'acc_peak_y_left', 'acc_peak_z_left', 'acc_peak_xyz_left', 'acc_peak_xy_left', 'acc_peak_yz_left', 'acc_peak_zx_left', 'gyro_mean_x_left', 'gyro_mean_y_left', 'gyro_mean_z_left', 'gyro_mean_xyz_left', 'gyro_mean_xy_left', 'gyro_mean_yz_left', 'gyro_mean_zx_left', 'gyro_std_x_left', 'gyro_std_y_left', 'gyro_std_z_left', 'gyro_std_xyz_left', 'gyro_std_xy_left', 'gyro_std_yz_left', 'gyro_std_zx_left', 'gyro_auc_x_left', 'gyro_auc_y_left', 'gyro_auc_z_left', 'gyro_auc_xyz_left', 'gyro_auc_xy_left', 'gyro_auc_yz_left', 'gyro_auc_zx_left', 'gyro_max_x_left', 'gyro_max_y_left', 'gyro_max_z_left', 'gyro_max_xyz_left', 'gyro_max_xy_left', 'gyro_max_yz_left', 'gyro_max_zx_left', 'gyro_peak_x_left', 'gyro_peak_y_left', 'gyro_peak_z_left', 'gyro_peak_xyz_left', 'gyro_peak_xy_left', 'gyro_peak_yz_left', 'gyro_peak_zx_left']

    #Shuffle the dataset
    adl = df.sample(n = len(df))
    #Reset shuffled index
    adl = adl.reset_index(drop = True)

    #Model Training
    x = adl[features]
    y = adl['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

    #activityTwo(x_train, x_test, y_train, y_test)

    #activityThree(x_train, x_test, y_train, y_test, x, y)

    #activityFour(x_train, y_train, adl)

    #activityFive(x, y, adl)

    activitySeven(x_train, x_test, y_train, y_test, x, y)

    #studio 4 activity 4
    #randomforest(x_train, y_train, x_test, y_test, x, y)#without adjusting the weights, oversampling performed
                                                        #a bit better, with zero false anythings
                                                        #undersampling had 13 false negatives
                                                        #Adjusting class weight did poorly, with 70 false negatives
                                                        #and 221 false positives
    #studio 4 activity 5
    df = pd.concat(map(pd.read_csv, ['ampc/w1.csv', 'ampc/w2.csv', 'ampc/w3.csv']), ignore_index = True)

    df.to_csv('studio4Activity5.csv')

    features = ['acc_mean_x_right', 'acc_mean_y_right', 'acc_mean_z_right', 'acc_mean_xyz_right', 'acc_mean_xy_right', 'acc_mean_yz_right', 'acc_mean_zx_right', 'acc_mean_pitch_right', 'acc_mean_roll_right', 'acc_std_x_right', 'acc_std_y_right', 'acc_std_z_right', 'acc_std_xyz_right', 'acc_std_xy_right', 'acc_std_yz_right', 'acc_std_zx_right', 'acc_std_pitch_right', 'acc_std_roll_right', 'acc_auc_x_right', 'acc_auc_y_right', 'acc_auc_z_right', 'acc_auc_xyz_right', 'acc_auc_xy_right', 'acc_auc_yz_right', 'acc_auc_zx_right', 'acc_auc_pitch_right', 'acc_auc_roll_right', 'acc_max_x_right', 'acc_max_y_right', 'acc_max_z_right', 'acc_max_xyz_right', 'acc_max_xy_right', 'acc_max_yz_right', 'acc_max_zx_right', 'acc_max_pitch_right', 'acc_max_roll_right', 'acc_peak_x_right', 'acc_peak_y_right', 'acc_peak_z_right', 'acc_peak_xyz_right', 'acc_peak_xy_right', 'acc_peak_yz_right', 'acc_peak_zx_right', 'gyro_mean_x_right', 'gyro_mean_y_right', 'gyro_mean_z_right', 'gyro_mean_xyz_right', 'gyro_mean_xy_right', 'gyro_mean_yz_right', 'gyro_mean_zx_right', 'gyro_std_x_right', 'gyro_std_y_right', 'gyro_std_z_right', 'gyro_std_xyz_right', 'gyro_std_xy_right', 'gyro_std_yz_right', 'gyro_std_zx_right', 'gyro_auc_x_right', 'gyro_auc_y_right', 'gyro_auc_z_right', 'gyro_auc_xyz_right', 'gyro_auc_xy_right', 'gyro_auc_yz_right', 'gyro_auc_zx_right', 'gyro_max_x_right', 'gyro_max_y_right', 'gyro_max_z_right', 'gyro_max_xyz_right', 'gyro_max_xy_right', 'gyro_max_yz_right', 'gyro_max_zx_right', 'gyro_peak_x_right', 'gyro_peak_y_right', 'gyro_peak_z_right', 'gyro_peak_xyz_right', 'gyro_peak_xy_right', 'gyro_peak_yz_right', 'gyro_peak_zx_right', 'acc_mean_x_left', 'acc_mean_y_left', 'acc_mean_z_left', 'acc_mean_xyz_left', 'acc_mean_xy_left', 'acc_mean_yz_left', 'acc_mean_zx_left', 'acc_mean_pitch_left', 'acc_mean_roll_left', 'acc_std_x_left', 'acc_std_y_left', 'acc_std_z_left', 'acc_std_xyz_left', 'acc_std_xy_left', 'acc_std_yz_left', 'acc_std_zx_left', 'acc_std_pitch_left', 'acc_std_roll_left', 'acc_auc_x_left', 'acc_auc_y_left', 'acc_auc_z_left', 'acc_auc_xyz_left', 'acc_auc_xy_left', 'acc_auc_yz_left', 'acc_auc_zx_left', 'acc_auc_pitch_left', 'acc_auc_roll_left', 'acc_max_x_left', 'acc_max_y_left', 'acc_max_z_left', 'acc_max_xyz_left', 'acc_max_xy_left', 'acc_max_yz_left', 'acc_max_zx_left', 'acc_max_pitch_left', 'acc_max_roll_left', 'acc_peak_x_left', 'acc_peak_y_left', 'acc_peak_z_left', 'acc_peak_xyz_left', 'acc_peak_xy_left', 'acc_peak_yz_left', 'acc_peak_zx_left', 'gyro_mean_x_left', 'gyro_mean_y_left', 'gyro_mean_z_left', 'gyro_mean_xyz_left', 'gyro_mean_xy_left', 'gyro_mean_yz_left', 'gyro_mean_zx_left', 'gyro_std_x_left', 'gyro_std_y_left', 'gyro_std_z_left', 'gyro_std_xyz_left', 'gyro_std_xy_left', 'gyro_std_yz_left', 'gyro_std_zx_left', 'gyro_auc_x_left', 'gyro_auc_y_left', 'gyro_auc_z_left', 'gyro_auc_xyz_left', 'gyro_auc_xy_left', 'gyro_auc_yz_left', 'gyro_auc_zx_left', 'gyro_max_x_left', 'gyro_max_y_left', 'gyro_max_z_left', 'gyro_max_xyz_left', 'gyro_max_xy_left', 'gyro_max_yz_left', 'gyro_max_zx_left', 'gyro_peak_x_left', 'gyro_peak_y_left', 'gyro_peak_z_left', 'gyro_peak_xyz_left', 'gyro_peak_xy_left', 'gyro_peak_yz_left', 'gyro_peak_zx_left']
    
    #Shuffle the dataset
    adl = df.sample(n = len(df))
    #Reset shuffled index
    adl = adl.reset_index(drop = True)

    #Model Training
    x = adl[features]
    y = adl['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

main()