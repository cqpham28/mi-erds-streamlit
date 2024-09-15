
"""
Data Formulation for FLEX DATASET

======================
Authors: Cuong Pham
cuongquocpham151@gmail.com

"""
import numpy as np
from moabb.paradigms import MotorImagery, FilterBankMotorImagery
from sklearn.preprocessing import LabelEncoder


################################
def _encode(y):
    """ encode label """
    le = LabelEncoder()
    return le.fit_transform(y)

################################
class Formulate():
    def __init__(
        self, 
        dataset = None, 
        fs = 128, 
        subject = 1, 
        bandpass = [[8,13]],
        channels = ("C3", "Cz", "C4"),
        t_rest = (-4,-2),
        t_mi = (0,2),
        run_to_split = None,
        ):
        """
        Usage:
            dataset = Flex2023_moabb()
            dataset.runs = 1
            f = Formulate(dataset, fs=128, subject=1, 
                        bandpass = [[8,13]],
                        channels = ("C3", "Cz", "C4"),
                        t_rest = (-4,-2),
                        t_mi = (0,2),
                        run_to_split=None,
                        )
            x, y = f.form(model_name="MI_2class_hand")
        """
        self.dataset = dataset
        self.fs = fs
        self.subject = subject
        self.bandpass = bandpass
        self.channels = channels
        self.t_rest = t_rest
        self.t_mi = t_mi
        self.run_to_split = run_to_split

        self.event_ids_4class = dict(
            right_hand=1, left_hand=2, right_foot=3, left_foot=4
        )

        self.event_ids_8class = dict(
            right_hand=1, left_hand=2, right_foot=3, left_foot=4,
            right_hand_r=5, left_hand_r=6, right_foot_r=7, left_foot_r=8
        )


    #-----------------------------------#
    def _extract_split_run(self, event_ids, interval):
        """ 
        Function to split run for data that has run combined (for F10, F11 only).
        Usage:
            if self.run_to_split is not None:
                x, y = self._extract_split_run(self.event_ids_hand, self.t_mi)
            else:
                x, y = self._extract("xy", self.event_ids_hand, self.t_mi)
        """

        x, y = self._extract("xy", self.event_ids_all, interval)

        if self.subject == 11: # 60trial x 3run
            split = [range(0, 60), range(60, 120), range(120, 180)]
        elif self.subject == 10: # 72trial x3run
            split = [range(0, 72), range(72, 144), range(144, 216)]

        # split for each run
        idx_r = self.run_to_split - 1
        x = x[split[idx_r]]
        y = y[split[idx_r]]

        # split for tasks
        idx_t = [i for i,v in enumerate(y) \
            if v in list(event_ids.keys())]
        x = x[idx_t]
        y = y[idx_t]

        return x, y


    #-----------------------------------#
    def _extract(self, returns:str, event_ids:dict, interval:tuple):
        """
        Get data/epochs
        """
        if self.bandpass is None:
            paradigm = MotorImagery(
                    events = list(event_ids.keys()),
                    n_classes = len(event_ids.keys()),
                    fmin = 0, 
                    fmax = self.fs/2-0.001, 
                    tmin = interval[0], 
                    tmax = interval[1], 
                    channels=self.channels,
                    resample=128,
                    )

        elif len(self.bandpass) == 1:
            paradigm = MotorImagery(
                    events = list(event_ids.keys()),
                    n_classes = len(event_ids.keys()),
                    fmin = self.bandpass[0][0], 
                    fmax = self.bandpass[0][1],
                    tmin = interval[0], 
                    tmax = interval[1], 
                    channels=self.channels,
                    resample=128,
                    )
        
        elif len(self.bandpass) > 1:
            paradigm = FilterBankMotorImagery(
                    filters=self.bandpass,
                    events = list(event_ids.keys()),
                    n_classes = len(event_ids.keys()),
                    tmin = interval[0],
                    tmax = interval[1],
                    channels=self.channels,
                    resample=128,
                    )

        if returns == "epochs":
            # do not use epochs.event in this case
            epochs,_,_ = paradigm.get_data(dataset=self.dataset,
                        subjects=[self.subject], return_epochs=True)
            return epochs

        elif returns == "xy":
            x,y,_ = paradigm.get_data(dataset=self.dataset,
                        subjects=[self.subject])
            return x, y


    #-----------------------------------#
    def form_4c_rest(self)->None:
        """ binary classifier: Rest/MI """

        x_rest,_ = self._extract("xy", self.event_ids_4class, self.t_rest)
        x_mi,_ = self._extract("xy", self.event_ids_4class, self.t_mi)

        y_rest = np.zeros(x_rest.shape[0])
        y_mi = np.ones(x_mi.shape[0])

        return np.concatenate((x_rest, x_mi)), np.concatenate((y_rest, y_mi))


    #-----------------------------------#
    def form_4c_all(self)->None:
        """ classifier: LH/RH/LF/RF (4class) """

        x, y = self._extract("xy", self.event_ids_4class, self.t_mi)
        return x, _encode(y)

    #-----------------------------------#
    def form_4c_2class_hand(self)->None:
        """ binary classifier (LH/RH) """

        event_ids = dict(right_hand=1, left_hand=2)
        x, y = self._extract("xy", event_ids, self.t_mi)
        return x, _encode(y)

    #-----------------------------------#
    def form_4c_2class_foot(self)->None:
        """ binary classifier (LF/RF) """

        event_ids = dict(right_foot=3, left_foot=4)
        x, y = self._extract("xy", event_ids, self.t_mi)
        return x, _encode(y)

    #-----------------------------------#
    def form_4c_2class_hand_foot(self)->None:
        """ binary classifier (LH+RH) & (LF+RF) """
        
        x1,_ = self._extract("xy", dict(right_hand=1, left_hand=2), self.t_mi)
        x2,_ = self._extract("xy", dict(right_foot=3, left_foot=4), self.t_mi)
        y1 = np.zeros(x1.shape[0],)
        y2 = np.ones(x2.shape[0],)
        
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        return x, _encode(y)


    #-----------------------------------#
    def form_4c_3class_rf(self)->None:
        """ classifier (LH/RH/RF) """

        event_ids = dict(right_hand=1, left_hand=2, right_foot=3)
        x, y = self._extract("xy", event_ids, self.t_mi)
        return x, _encode(y)

    #-----------------------------------#
    def form_4c_3class_lf(self)->None:
        """ classifier (LH/RH/LF) """

        event_ids = dict(right_hand=1, left_hand=2, left_foot=4)
        x, y = self._extract("xy", event_ids, self.t_mi)
        return x, _encode(y)
    

    #-----------------------------------#
    def form_8c_rest(self, t_rest=(2.5, 4.5))->None:
        """ get data REST (using t_rest in the final 3s of trial.)"""

        x,_ = self._extract("xy", self.event_ids_8class, t_rest)
        y = ["rest" if "_r" in i else "no_rest" for i in _]
        return x, _encode(y)


    #-----------------------------------#
    def form_8c_rest_base(self)->None:
        """ get data REST BASE (using default t_rest before cue)"""

        # REST BASE (using t_rest before cue)
        x_mi,_ = self._extract("xy", self.event_ids_8class, (0,2))
        x_rest,_ = self._extract("xy", self.event_ids_8class, (-4,-2))
        y_rest = np.zeros(x_rest.shape[0])
        y_mi = np.ones(x_mi.shape[0])
        x = np.concatenate((x_rest, x_mi))
        y = np.concatenate((y_rest, y_mi))

        return x, y

    #-----------------------------------#
    def form_8c_mi(self)->None:
        """ get data for MI-4class model in 8c protocol """

        x, y_global  = self._extract("xy", self.event_ids_8class, (0, 2))
        y = [i[:-2] if "_r" in i else i for i in y_global]
        return x, _encode(y)


    #-----------------------------------#
    def form_8c_hand(self)->None:
        """ get data for LH-RH model in 8c protocol """

        event_ids = dict(
            right_hand=1, left_hand=2,
            right_hand_r=5, left_hand_r=6,
        )
        x, y_global  = self._extract("xy", event_ids, (0, 2))
        y = [i[:-2] if "_r" in i else i for i in y_global]
        return x, _encode(y)
    
    #-----------------------------------#
    def form_8c(self, t_rest=(2.5, 4.5))->None:
        """ get data for combined validation"""
        # MI
        x1, y_global  = self._extract("xy", self.event_ids_8class, (0, 2))
        y1 = [i[:-2] if "_r" in i else i for i in y_global]
        le1 = LabelEncoder(); y1 = le1.fit_transform(y1)

        ## REST
        x2, _ = self._extract("xy", self.event_ids_8class, t_rest)
        y2 = ["rest" if "_r" in i else "no_rest" for i in y_global]
        le2 = LabelEncoder(); y2 = le2.fit_transform(y2)

        # ## debug
        # for i in range(len(y_global)):
        #     print(f"index {i} | global: {y_global[i]}, local1: {y1[i]}, local2: {y2[i]}")
        
        # fix
        y2 = y2.reshape(-1,1) # for binary
        x1 = x1[:,:,:-1]
        x2 = x2[:,:,:-1]

        return (x1,y1,x2,y2, y_global, le1, le2)



    #-----------------------------------#
    def form(self, model_name:str) -> None:
        """ caller """

        if model_name == "4c_rest":
            x, y = self.form_4c_rest()

        elif model_name == "4c_2class_handfoot":
            x, y = self.form_4c_2class_hand_foot()

        elif model_name == "4c_2class_hand":
            x, y = self.form_4c_2class_hand()
        
        elif model_name == "4c_2class_foot":
            x, y = self.form_4c_2class_foot()
            
        elif model_name == "4c_3class_lf":
            x, y = self.form_4c_3class_lf()

        elif model_name == "4c_3class_rf":
            x, y = self.form_4c_3class_rf()

        elif model_name == "4c_all":
            x, y = self.form_4c_all()

        #------#
        elif model_name == "8c_rest":
            x, y = self.form_8c_rest()

        elif model_name == "8c_mi":
            x, y = self.form_8c_mi()

        elif model_name == "8c_hand":
            x, y = self.form_8c_hand()
        
        else:
            raise ValueError(f"model_name {model_name} is not supported")
        
        print(f"\n({model_name}) | x: {x.shape}, y: {y.shape}")
        a,b = np.unique(y, return_counts=True)
        print(f"({model_name}) | unique: {[(i,v) for (i,v) in zip(a,b)]}")
        
        # # if binary, reshape label
        # if len(a)==2:
        #     y = y.reshape(-1,1)

        return x[:,:,:-1], y

	
