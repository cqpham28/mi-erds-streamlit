"""
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from moabb.pipelines.utils import FilterBank
from moabb.paradigms import MotorImagery, FilterBankMotorImagery
from mne.decoding import CSP

from src.data import Flex2023_moabb_st 



################################
class Formulate():
	def __init__(
        self, 
        dataset, 
        fs, 
        subject, 
        feature="csp",
        channels=("C3", "Cz", "C4"),
        t_rest=(-2,0),
        t_mi=(0,2)
        ):
		"""
		Args:
			dataset: moabb dataset
			fs (float):
			subject (int):

		"""

		self.dataset = dataset
		self.subject = subject

		self.channels = channels
		self.feature = feature
		self.t_rest = t_rest
		self.t_mi = t_mi
		self.event_ids_all = dataset.event_id
		self.event_ids = {k:v for (k,v) in dataset.event_id.items() \
				if k in ["left_hand", "right_hand"]}
		
		# only for Flex2023
		self.event_ids_foot = {k:v for (k,v) in dataset.event_id.items() \
				if k in ["left_foot", "right_foot"]}
		self.fs = fs

		
	def _extract(self, returns:str, event_ids:dict, interval:tuple):
		"""
		Get data/epochs depends on specific t_interval
		Args:
			interval (tuple) 
		"""
		if self.feature == "csp":
			paradigm = MotorImagery(
					events = list(event_ids.keys()),
					n_classes = len(event_ids.keys()),
					# fmin = 0, fmax = self.fs/2-0.001, 
					fmin = 8, fmax = 13, 
					tmin = interval[0], 
					tmax = interval[1], 
					channels=self.channels,
					resample=128,
					)
		
		elif self.feature == "fbcsp":
			filters = ([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32])
			paradigm = FilterBankMotorImagery(
					filters=filters,
					events = list(event_ids.keys()),
					n_classes = len(event_ids.keys()),
					tmin = interval[0],
					tmax = interval[1],
					channels=self.channels,
					resample=128,
					)
	
		if returns == "epochs":
			epochs,_,_ = paradigm.get_data(dataset=self.dataset,
					subjects=[self.subject], return_epochs=True)
			return epochs

		elif returns == "xy":
			x,y,_ = paradigm.get_data(dataset=self.dataset,
					subjects=[self.subject])
			return x, y

	

	def form_binary(self)->None:
		""" get data for Rest/NoRest classifier"""

		x_rest, _ = self._extract("xy", self.event_ids_all, self.t_rest)
		x_mi, _ = self._extract("xy", self.event_ids_all, self.t_mi)

		y_rest = np.zeros(x_rest.shape[0])
		y_mi = np.ones(x_mi.shape[0])

		x = np.concatenate((x_rest, x_mi))
		y = np.concatenate((y_rest, y_mi))

		return x, y
	

	def form_mi_all(self)->None:
		""" get data for MI model 
		(to be cross-check with moabb evaluation)"""

		if self.feature == "csp":
			epochs = self._extract("epochs", self.event_ids_all, self.t_mi)
			x = epochs.get_data(units='uV')
			y = epochs.events[:,-1]
		
		elif self.feature == "fbcsp":
			x, y = self._extract("xy", self.event_ids_all, self.t_mi)
			le = LabelEncoder()
			y = le.fit_transform(y)

		return x, y
	

	def form_mi_2class(self)->None:
		""" get data for MI model 
		(to be cross-check with moabb evaluation)"""

		x, y = self._extract("xy", self.event_ids, self.t_mi)
		le = LabelEncoder()
		y = le.fit_transform(y)
		return x, y


	def form_mi_2class_foot(self)->None:
		""" get data for MI model 
		(to be cross-check with moabb evaluation)"""

		x, y = self._extract("xy", self.event_ids_foot, self.t_mi)
		le = LabelEncoder()
		y = le.fit_transform(y)
		return x, y


	
	def form(self, model_name=""):
		""" formulate data """

		if model_name == "MI_2class_hand":
			x, y = self.form_mi_2class()
		elif model_name == "MI_2class_foot":
			x, y = self.form_mi_2class_foot()
		elif model_name == "MI_all":
			x, y = self.form_mi_all()
		elif model_name == "Rest/NoRest":
			x, y = self.form_binary()
			
		return x, y



################################
class Pipeline_CSP_Scratch():
	"""
	Run pipeline for (x,y), either one of 3 models 
		MI_2class / MI_all / Rest_NoRest
	"""
	def __init__(self, feature="csp", classifier="lda"):


		if feature == "csp":
			self.fe = CSP(n_components=8)
		elif feature == "fbcsp":
			self.fe = FilterBank(estimator=CSP(n_components=4))

		if classifier == "lda":
			self.classifier = LDA()
		elif classifier == "svm":
			self.classifier = SVC(C=5, 
							gamma='auto', 
							kernel='rbf',
							probability=True, 
							random_state=42),

	
	def _pipe(self, x_train, y_train, x_test, y_test):
		""" 
		fit for 1 fold. return:
			pipe_ml: trained model on train-set
			r: metrics on test set (binary-> auc_roc | multiclass -> accuracy)
		"""

		pipe_process = make_pipeline(self.fe)
		pipe_process.fit(x_train, y_train)
		ft_train = pipe_process.transform(x_train) # (N,chan,times)->(N,chan)
		ft_test = pipe_process.transform(x_test)

		## classifier
		pipe_ml = Pipeline(
			steps=[('classifier', self.classifier)
			])
		
		# train
		pipe_ml.fit(ft_train, y_train)

		## test
		if len(np.unique(y_train))==2: # binary
			r = metrics.roc_auc_score(y_test, 
						pipe_ml.predict_proba(ft_test)[:, 1])
		else: # multiclass
			r = metrics.accuracy_score(y_test, 
						pipe_ml.predict(ft_test))

		return pipe_ml, r


	def run(self, x, y):
		""" kfold cross cv"""

		df = pd.DataFrame()
		skf = StratifiedKFold(n_splits=5,
							shuffle=True,
							random_state=42)

		for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
			pipe, r = self._pipe(x[train_index],
								y[train_index],
								x[test_index],
								y[test_index],
								)
			df.loc[f"fold-{fold}", "score"] = r
		
		return df
			





################################
def run_ml_feedback():
	""" machine learning pipeline """


	## ADAPT STREAMLIT
	subject = st.session_state.current_subject
	run = st.session_state.current_run

	dataset = Flex2023_moabb_st()
	fs = 128
	dataset.subject_list = [subject]
	dataset.runs = run

	df = pd.DataFrame()
	j = 0
	
	for t_mi in [(-2,0), (-1,1), (0,2), (1,3), (2,4)]:
		for feature in ["csp"]:
			for classifier in ["lda"]:
				for model_name in ["MI_2class_hand", "MI_2class_foot", "MI_all"]:
					channels = ("C3", "Cz", "C4")
					# data
					f = Formulate(dataset, fs, subject, 
								feature=feature,
								channels=channels,
								t_rest=(-4,-2),
								t_mi=t_mi
								)
					x, y = f.form(model_name)
					_,count = np.unique(y, return_counts=True)

					#
					p = Pipeline_CSP_Scratch(feature, classifier)
					_df = p.run(x, y)

					# save
					df.loc[j, "subject"] = subject
					df.loc[j, "runs"] = run
					df.loc[j, "trials/class"] = f"{count.mean():.1f}"
					df.loc[j, "t_mi"] = str([j+4 for j in t_mi]) # the event start at 4s
					df.loc[j, "method"] = f"{feature}+{classifier}"
					df.loc[j, "model_name"] = model_name
					df.loc[j, "channels"] = str(channels)
					df.loc[j, "score"] = _df['score'].mean()
					j += 1

	return df




def plot_ml_feedback(df):
	""" visualization """

	g = sns.catplot(
		data=df,
		x="t_mi",
		y="score",
		hue="model_name",
		row="method",
		kind="bar",
		palette="viridis", 
		height=3.5, aspect=3,
	)
	# iterate through axes
	for ax in g.axes.ravel():
		# add annotations
		for c in ax.containers:
			labels = [f"{v.get_height():.2f}" for v in c]
			ax.bar_label(c, labels=labels, label_type='edge')
		ax.margins(y=0.2)

	plt.ylim((0,1))
	# plt.suptitle(f"MI_2class (3chan, t=0-3s)", fontweight='bold', x=0.9)
	# plt.show()

	# plt.tight_layout()
	plt.savefig(f'refs/catplot.png')
	img = Image.open(f'refs/catplot.png')

	return img
