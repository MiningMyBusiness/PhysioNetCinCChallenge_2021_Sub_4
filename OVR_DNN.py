import numpy as np
import pandas as pd
import pickle
import glob 
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from imblearn.combine import SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier

from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler 

from timeit import default_timer

import glob

class OVR_DNN:
    
    def __init__(self, X_train=None, y_train=None, filename=None):
        self._X_train = X_train
        self._y_train = y_train
        self._filename = filename
        if self._X_train is not None:
            self.find_imp_feats()
            self.train_base_models()
            self.train_stk_models()
        if self._filename is not None:
            print('Loading models...')
            self.load_models()
        
        
    def find_imp_feats(self):    
        self.split_data()
        estimators = self.get_models()
        first_one = True
        feat_imp_sums = np.zeros(self._X_train.shape[1])
        for pair in estimators:
            print('Training base model to find feature importances', pair[0])
            pair[1].fit(self._X_train, self._y_train)
            for est in pair[1].estimators_:
                try:
                    if hasattr(est[1], 'feature_importances_'):
                        print('Found estimator with feature importances!...')
                        feat_imp_sums += est[1].feature_importances_
                except:
                    print('DOES NOT HAVE FEATURE IMPORTANCES')
        self._imp_feats = feat_imp_sums > np.mean(feat_imp_sums)
        self._X_train = self.limit_to_imp_feats(self._X_train)
        self._X_val = self.limit_to_imp_feats(self._X_val)
    
    
    
    def train_base_models(self):
        estimators = self.get_models()
        first_one = True
        for pair in estimators:
            print('Training base model with important features', pair[0])
            pair[1].fit(self._X_train, self._y_train)
        self._base_models = estimators
        
        
        
        
        
    def get_models(self):
        base_lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced'))
        ovr_lr = OneVsRestClassifier(base_lr)

        base_rf = make_pipeline_with_sampler(RandomUnderSampler(), RandomForestClassifier(n_jobs=-1))
        ovr_rf = OneVsRestClassifier(base_rf)

        base_et = make_pipeline_with_sampler(RandomUnderSampler(), ExtraTreesClassifier(n_jobs=-1))
        ovr_et = OneVsRestClassifier(base_et)

        base_gbc = make_pipeline_with_sampler(RandomUnderSampler(), HistGradientBoostingClassifier())
        ovr_gbc = OneVsRestClassifier(base_gbc)

        estimators = [('lr', ovr_lr),
              ('rf', ovr_rf),
              ('et', ovr_et),
              ('gbc', ovr_gbc)]
        return estimators
    
    
    def split_data(self):
        print('Splitting data into training and validation set to train DNNs...')
        X_train, y_train, X_val, y_val = iterative_train_test_split(self._X_train, 
                                                              self._y_train, 
                                                              test_size = 0.35)
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        print('Train data:', X_train.shape)
        print('Train labels:', y_train.shape)
        print('Val data:', X_val.shape)
        print('Val labels:', y_val.shape)
            
            
    def train_stk_models(self):
        print('Training stacking model on validation set...')
        for i,model in enumerate(self._base_models):
            print('  Getting probabilities for validation set...')
            this_y_prob = model[1].predict_proba(self._X_val)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)
                
        stk_et = make_pipeline_with_sampler(RandomUnderSampler(), ExtraTreesClassifier(n_jobs=-1))
        ovr_stk_et = OneVsRestClassifier(stk_et)
        ovr_stk_et.fit(y_prob, self._y_val)
        self._stk_model = ovr_stk_et
    
    
    
    def predict(self, X_test):
        X_test = self.limit_to_imp_feats(X_test)
        for i,model in enumerate(self._base_models):
            this_y_prob = model[1].predict_proba(X_test)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)
        
        y_pred_test = self._stk_model.predict(y_prob)
        return y_pred_test
    
    
    
    def predict_proba(self, X_test):
        X_test = self.limit_to_imp_feats(X_test)
        for i,model in enumerate(self._base_models):
            this_y_prob = model[1].predict_proba(X_test)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)
        
        y_pred_test = self._stk_model.predict_proba(y_prob)
        return y_pred_test
    
    
    
    def save_models(self, filename):
        model_dict = {
            'base_models': self._base_models,
            'imp_feats': self._imp_feats,
            'stk_model': self._stk_model
        }
        stk_filename = filename.split('.pick')[0] + '_ovr_imb_models.pickle'
        with open(stk_filename, 'wb') as handle:
            pickle.dump(model_dict, handle)
        handle.close()
        
    
    
    
    def load_models(self):
        # load in base dnn models
        file_hash = self._filename.split('.pick')[0] + '_ovr_imb_models.pickle'
        with open(file_hash, 'rb') as handle:
            model_dict = pickle.load(handle)
        handle.close()
        self._base_models = model_dict['base_models']
        self._imp_feats = model_dict['imp_feats']
        self._stk_model = model_dict['stk_model']
        
        
    def limit_to_imp_feats(self, X):
        X = X[:, self._imp_feats]
        return X
    
    
    