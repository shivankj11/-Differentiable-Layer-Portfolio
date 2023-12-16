from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
import matplotlib.pyplot as plt
class Backtesting:
    def __init__(self,experiment_name=EXP_NAME,is_weight=False,is_pred=False,weight=None,pred=None):
        self.experiment_name = experiment_name
        if is_weight:
            self.weight = weight
        if is_pred:
            self.pred = pred
        if self.weight is None or self.pred is None:
            raise ValueError("weight and pred must be provided")

        self.ready_weightonly = False
        self.ready_predonly = False
        self.ready_oneovern = False
        self.ready_opt = False
        self.plotinfo = {}
        self.standard_plot_name = ["pnl_curve","endtime_CEQ",]#"avg_CEQ"]

    def SetupData(self,tc,instructment):
        self.iu = instructment
        handler_kwargs2 = {
            "start_time": tc["all_start"],
            "end_time": tc["all_end"],
            "instruments": instructment,
            "data_loader":QlibDataLoader(config={"feature":["Ref($close,-2)","Ref($close,-1)"]}),
            "infer_processors":[
            {"class": "DropnaProcessor" ,"kwargs":{"fields_group": "feature"}},]
            
        }
        hd2 = DataHandlerLP(**handler_kwargs2)
        dataset_conf2 = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": hd2,
                "segments": {
                    "train": (tc["train_start"], tc["train_end"]),
                    "valid": (tc["valid_start"], tc["valid_end"]),
                    "test": (tc["test_start"], tc["test_end"]),
                },
            },
        }
        self.ds = init_instance_by_config(dataset_conf2).prepare("test", data_key=DataHandlerLP.DK_I)
    def _search_record(self,rid,name="pred.pkl"):
        rec = R.get_recorder(experiment_name=self.experiment_name,recorder_id=rid)
        return  rec.load_object(name)
    def _BacktestForWeightOnly(self):
        print("backtesting for direct weight")
        pred_rec = self._search_record(rid=self.weight)
        print("-info::",pred_rec)
        btdata = self.ds
        breakthres =20000
        values = []
        cumvalue = []
        asset = [1]
        initial_close = None
        CEQ = nms.get_objective_func()
        for (dt,df) in btdata.groupby("datetime"):
            breakthres -= 1
            if breakthres == 0:
                break

            close_price2 = df["Ref($close,-2)"].values
            close_price1 = df["Ref($close,-1)"].values
            if len(close_price2) != len(iu):
                continue
            if type(initial_close) == type(None):
                initial_close = close_price1
            weight = pred_rec.groupby("datetime").get_group(dt)["score"].values
            # calculate return from close_price1 to close_price2
            return1 = close_price2/close_price1 - 1
            # check if return1 has nan
            if np.isnan(return1).any():
                continue
            #print("debug ",return1,weight)
            asset.append(asset[-1] * (1+return1.dot(weight)))
            # check if asset[-1] is nan
            if np.isnan(asset[-1]):
                print("asset,return1,close1,close2,weight",asset,return1,close_price1,close_price2,weight,"return * weight=",(1+return1).dot(weight))
                break
            values.append(-CEQ(return1,weight))
            cumvalue.append(-CEQ(close_price2/initial_close-1,weight))   
        standard_plot = {}
        standard_plot["pnl_curve"] = asset
        standard_plot["endtime_CEQ"] = cumvalue
        # cum prefix sum of values
        print(values)
        standard_plot["avg_CEQ"] = list(np.cumsum(np.array(values)))
        self.ready_weightonly = True
        self.plotinfo["directweight"] = standard_plot
    def _BacktestForPredOnly(self):
        print("backtesting for pred2weight")
        pred_rec = self._search_record(rid=self.pred)["score"]
        # only remain instructment in self.iu row in pred_rec
        pred_rec = pred_rec[pred_rec.index.get_level_values(1).isin(self.iu)]
        btdata = self.ds
        breakthres =20000
        values = []
        cumvalue = []
        asset = [1]
        initial_close = None
        CEQ = nms.get_objective_func()
        for (dt,df) in btdata.groupby("datetime"):
            breakthres -= 1
            if breakthres == 0:
                break

            close_price2 = df["Ref($close,-2)"].values
            close_price1 = df["Ref($close,-1)"].values
            if len(close_price2) != len(iu):
                continue
            if type(initial_close) == type(None):
                initial_close = close_price1
            pred_day = pred_rec.groupby("datetime").get_group(dt).values
            weight = nms(torch.from_numpy(pred_day)).detach().numpy()
            
            # calculate return from close_price1 to close_price2
            return1 = close_price2/close_price1 - 1
            # check if return1 has nan
            if np.isnan(return1).any():
                continue
            asset.append(asset[-1] * (1+return1.dot(weight)))
            # check if asset[-1] is nan
            if np.isnan(asset[-1]):
                print("asset,return1,close1,close2,weight",asset,return1,close_price1,close_price2,weight,"return * weight=",(1+return1).dot(weight))
                break
            # print("debug", return1, weight)
            values.append(-CEQ(return1,weight))
            cumvalue.append(-CEQ(close_price2/initial_close-1,weight))   
        standard_plot = {}
        standard_plot["pnl_curve"] = asset
        standard_plot["endtime_CEQ"] = cumvalue
        # cum prefix sum of values
        print(values)
        standard_plot["avg_CEQ"] = list(np.cumsum(np.array(values)))
        self.ready_predonly= True
        self.plotinfo["pred2weight"] = standard_plot
        
    def _BacktestForOneOverN(self):
        print("backtesting for 1 over N")
        btdata = self.ds
        breakthres =20000
        values = []
        cumvalue = []
        asset = [1]
        initial_close = None
        CEQ = nms.get_objective_func()
        for (dt,df) in btdata.groupby("datetime"):
            breakthres -= 1
            if breakthres == 0:
                break

            close_price2 = df["Ref($close,-2)"].values
            close_price1 = df["Ref($close,-1)"].values
            if len(close_price2) != len(iu):
                continue
            if type(initial_close) == type(None):
                initial_close = close_price1
            weight = np.ones(len(iu))  * 1.0/len(iu)
            # calculate return from close_price1 to close_price2
            return1 = close_price2/close_price1 - 1
            # check if return1 has nan
            if np.isnan(return1).any():
                continue
            asset.append(asset[-1] * (1+return1.dot(weight)))
            # check if asset[-1] is nan
            if np.isnan(asset[-1]):
                print("asset,return1,close1,close2,weight",asset,return1,close_price1,close_price2,weight,"return * weight=",(1+return1).dot(weight))
                break
            # print("debug", return1, weight)
            values.append(-CEQ(return1,weight))
            cumvalue.append(-CEQ(close_price2/initial_close-1,weight))   
        standard_plot = {}
        standard_plot["pnl_curve"] = asset
        standard_plot["endtime_CEQ"] = cumvalue
        # cum prefix sum of values
        print(values)
        standard_plot["avg_CEQ"] = list(np.cumsum(np.array(values)))
        self.ready_oneovern= True
        self.plotinfo["OneOverN"] = standard_plot
        

    def _BacktestOpt(self):
        print("backtesting for opt")
        btdata = self.ds
        breakthres =20000
        values = []
        cumvalue = []
        asset = [1]
        initial_close = None
        CEQ = nms.get_objective_func()
        for (dt,df) in btdata.groupby("datetime"):
            breakthres -= 1
            if breakthres == 0:
                break

            close_price2 = df["Ref($close,-2)"].values
            close_price1 = df["Ref($close,-1)"].values
            if len(close_price2) != len(iu):
                continue
            if type(initial_close) == type(None):
                initial_close = close_price1
            # calculate return from close_price1 to close_price2
            returni = close_price2/initial_close-1
            return1 = close_price2/close_price1 - 1
            weight = nms(torch.from_numpy(returni)).detach().numpy()
            
            # check if return1 has nan
            if np.isnan(return1).any():
                continue
            asset.append(asset[-1] * (1+return1.dot(weight)))
            # check if asset[-1] is nan
            if np.isnan(asset[-1]):
                print("asset,return1,close1,close2,weight",asset,return1,close_price1,close_price2,weight,"return * weight=",(1+return1).dot(weight))
                break
            # print("debug", return1, weight)
            values.append(-CEQ(return1,weight))
            cumvalue.append(-CEQ(close_price2/initial_close-1,weight))   
        standard_plot = {}
        standard_plot["pnl_curve"] = asset
        standard_plot["endtime_CEQ"] = cumvalue
        # cum prefix sum of values
        print(values)
        standard_plot["avg_CEQ"] = list(np.cumsum(np.array(values)))
        self.ready_opt= True
        self.plotinfo["opt"] = standard_plot
    
    def ExecuteAll(self):
        self._BacktestForPredOnly()
        self._BacktestForWeightOnly()
        self._BacktestForOneOverN()
        self._BacktestOpt()
    def ShowAll(self):
        for key in self.standard_plot_name:
            for k,v in self.plotinfo.items():
                # color illustration
                
                plt.plot(v[key])
            plt.title(key)
            plt.show()

            
    
    def ShowOne(self,which):
        sp = self.plotinfo[which]
        for k in self.standard_plot_name:
            plt.plot(sp[k])
            plt.title(which+" "+ k)
            plt.show()