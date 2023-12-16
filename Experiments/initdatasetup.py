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
