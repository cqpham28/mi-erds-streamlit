"""
Machine Learning Pipeline
"""

import numpy as np
import pandas as pd
from moabb.pipelines.utils import FilterBank
from mne.decoding import CSP

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import Lasso
# import mrmr
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
#
from src.config import *

ALPHA_LASSO_SPARSE = 0.05



#==============================#
class METHODS:
    pipeline = {

        "CSP+LDA": Pipeline(steps=[
                ('csp', CSP(n_components=8)),
                ('lda', LDA())]
            ),
        
        "alphaCSP+CCA+LDA": None,
        "FBCSP+LDA": None,
        "SparseFBCSP+LDA": None,

        # #--------#
        "Cov+Tangent+SVM": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('tangent_Space', TangentSpace(metric="riemann")),
                ('svm', SVC(kernel='linear', random_state=SEEDS, probability=True))]
            ),

        "Cov+Tangent+LDA": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('tangent_Space', TangentSpace(metric="riemann")),
                ('lda', LDA())],
            ),

        "Cov+Tangent+LR": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('TS', TSclassifier())]
            ),

        "Cov+MDM": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('mdm', MDM(metric=dict(mean='riemann', distance='riemann')))]
            ),
            
    }


#==============================#
class Pipeline_ML():
    """
    Run pipeline for (x,y), either one of 3 models 
        MI_2class / MI_all / Rest_NoRest
    """
    def __init__(self, method="CSP+LDA"):
        self.method = method

    #---------------------------------------#
    def _pipe(self, x_train, y_train, x_test, y_test):
        """ 
        fit for 1 fold. return:
            pipe_ml: trained model on train-set
            r: metrics on test set (binary-> auc_roc | multiclass -> accuracy)
        """
        #------------#
        if "FBCSP" in self.method:
            fbcsp = make_pipeline(FilterBank(CSP(n_components=4)))
            fbcsp.fit(x_train, y_train)
            ft_train = fbcsp.transform(x_train)
            ft_test = fbcsp.transform(x_test) 

            if "Sparse" in self.method:
                # Apply sparse feature selection using Lasso
                clf = Lasso(alpha=ALPHA_LASSO_SPARSE,
                            random_state=SEEDS,
                            selection="random")

                clf.fit(ft_train, y_train)
                idx = [i for i,v in enumerate(clf.coef_) if v!=0]
                ft_train_select = ft_train[:, idx]
                ft_test_select = ft_test[:, idx]
            
            
            else:
                ft_train_select = ft_train
                ft_test_select = ft_test

            ## train
            pipe_ml = Pipeline(steps=[('classifier', LDA())])
            pipe_ml.fit(ft_train_select, y_train)
            ## test
            if len(np.unique(y_train))==2: # binary
                r = metrics.roc_auc_score(y_test, 
                            pipe_ml.predict_proba(ft_test_select)[:, 1])
            else: # multiclass
                r = metrics.accuracy_score(y_test, 
                            pipe_ml.predict(ft_test_select))
        
        #------------#
        else:
            ## train
            pipe_ml = METHODS.pipeline[self.method]
            pipe_ml.fit(x_train, y_train)
            ## test
            if len(np.unique(y_train))==2: # binary
                r = metrics.roc_auc_score(y_test, 
                            pipe_ml.predict_proba(x_test)[:, 1])
            else: # multiclass
                r = metrics.accuracy_score(y_test, 
                            pipe_ml.predict(x_test))

        return pipe_ml, r


    #---------------------------------------#
    def run(self, x, y, augment=False):
        """ kfold cross cv"""
        
        # for augment
        step = 256 
        overlap = 64
        size = 512

        df = pd.DataFrame()
        skf = StratifiedKFold(n_splits=KFOLD,
                            shuffle=True,
                            random_state=SEEDS)

        for fold, (train_index, test_index) in enumerate(skf.split(x, y)):

            if augment:
            
                x_train = [x[train_index,:, i:i+step] for i in range(0, size-step, overlap)]
                y_train = [y[train_index] for i in range(0, size-step, overlap)]
                x_train = np.concatenate(x_train)
                y_train = np.concatenate(y_train)

                x_test = x[test_index,:,:step]
                y_test = y[test_index]
                
                print(x_train.shape)
                print(y_train.shape)
                print(x_test.shape)
                print(y_test.shape)

                pipe, r = self._pipe(x_train, y_train, x_test, y_test)
            
            else:

                pipe, r = self._pipe(x[train_index],
                                        y[train_index],
                                        x[test_index],
                                        y[test_index],
                )
            
            ## log
            df.loc[f"fold-{fold}", "score"] = r
        
        return df