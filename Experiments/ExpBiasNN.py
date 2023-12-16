import sys
print(sys.path)
# get the parent directory of the current script
import os
cwd = os.path.abspath(os.path.join(os.getcwd(), "../"))
print("cwd:",cwd)
# insert cwd/+"../qlib" into sys.path as qlib
sys.path.insert(0, cwd + "/qlib")

from pprint import pprint
from pathlib import Path
import pandas as pd
MARKET = "csi300"
BENCHMARK = "SH000300"
EXP_NAME = "ExpBIASNN"
import qlib
print("qlib:",qlib.__path__)
from qlib.tests.data import GetData

GetData().qlib_data(exists_skip=True)


qlib.init()
from qlib.data import D
p = Path(cwd+"/../data/qlib_data/cn_data/financial").expanduser()


iu = ["SH601318", "SZ000651",  "SH600519", "SZ000002", "SH600276",   "SH600028", "SZ000063",]

from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158

handler_kwargs = {
    "start_time": "2008-01-01",
    "end_time": "2020-03-01",
    "fit_start_time": "2010-01-01",
    "fit_end_time": "2020-12-31",
    "instruments": iu,
    "infer_processors": [
        {"class": "FilterCol",
          "kwargs":{
              "fields_group": "feature",
              "col_list": ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", 
                            "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", 
                            "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"
                        ]
            }
        },
        {"class": "RobustZScoreNorm",
          "kwargs":{
              "fields_group": "feature",
              "clip_outlier": True
            }
        },
        {"class": "Fillna",
          "kwargs":{
              "fields_group": "feature"}}],
    "learn_processors": [{
        "class": "DropnaLabel"},
        {"class": "CSRankNorm",
          "kwargs":{
              "fields_group": "label"}}
    ],
    "label":["Ref($close, -2) / Ref($close, -1) - 1"]
}
hd = Alpha158(**handler_kwargs)
dataset_conf = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": hd,
        "segments": {
            "train": ("2010-01-01", "2016-6-30"),
            "valid": ("2016-07-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-01-01"),
        },
        "step_len":20
    },
}
dataset1 = init_instance_by_config(dataset_conf)

# read ../data/corr_cvxpylayer.pkl
import pickle
with open("../data/corr_cvxpylayer.pkl",'rb') as f:
    cov = pickle.load(f)
# use iu list to select corr submatrix
cov = cov.loc[iu,iu]
print(cov.shape)
print(cov)
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
rid = "49f75616f8a349f099c31db78a102722"
rec = R.get_recorder(experiment_name=EXP_NAME,recorder_id=rid)
pred_model = rec.load_object("trained_model")
pred_res = rec.load_object("pred.pkl")
# sort pred_res by datetime, pred_res is pd, datetime is its col
pred_res = pred_res.sort_values(by=['datetime'])
print(pred_res)
# we use sample cov as stated in markowitz 1952
from qlib.utils import NumericalMarkowitzSolverLayer
import numpy as np 
import torch
cov_sqrt = pd.DataFrame(np.linalg.cholesky(cov),index=cov.index,columns=cov.columns)
# typical risk aversion is from 1 to 10 https://quant.stackexchange.com/questions/8405/typical-risk-aversion-parameter-value-for-mean-variance-optimization
nms = NumericalMarkowitzSolverLayer(torch.tensor(cov_sqrt.values).float(),torch.tensor([1.0]).float(),0,len(iu))

from qlib.contrib.model.cvxpizer import *
import torch
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.utils import init_instance_by_config
model = init_instance_by_config(
    {
        "class": "AuxNN",
        "module_path": "qlib.contrib.model.pytorch_auxNN",
        "module_path": "qlib.contrib.model.pytorch_auxNN",
        "kwargs": {
            "d_feat": 20,
            "instructment_universe": iu,
            "decision_layer": nms,
            "pred_model": pred_model.GRU_model,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 50,
            "auxlr": 1e-4,
            "dfusionlr": 1e-6,
            "early_stop": 10,
            "batch_size": 800,
            "metric": "loss",
            "loss": "mse",
            "n_jobs": 20,
            "GPU": 2,
            "verbose": True,
            "regulator_aversion":5.0
        },
    }
)

# start exp to train model
with R.start(experiment_name=EXP_NAME):
    model.fit(dataset1)
    R.save_objects(trained_model=model)

    rec = R.get_recorder()
    rid = rec.id  # save the record id

    # Inference and saving signal
    sr = SignalRecord(model, dataset1, rec)
    sr.generate()
