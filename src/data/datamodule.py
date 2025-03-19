import warnings
import torch
from torch.utils.data import ConcatDataset, DataLoader
from src.data.batch_scheduler import BatchSchedulerSampler
from src.data.random_batch_scheduler_1channel import BatchSchedulerSampler_1channel
from src.data.pred_dataset import *


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        
        if "split" in dataset_kwargs.keys():
           del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val
    
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
        
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):        
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0: return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # get batch_size if not defined
        if batch_size is None: batch_size=self.batch_size        
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data

DSETS = {'ETTm1':Dataset_ETT_minute,
         'ETTm2':Dataset_ETT_minute,
         'ETTh1':Dataset_ETT_hour, 
         'ETTh2':Dataset_ETT_hour, 
         'electricity':Dataset_Custom,
         'traffic':Dataset_Custom,
         'illness':Dataset_Custom, 
         'weather':Dataset_Custom, 
         'exchange':Dataset_Custom,
         'nn5':Dataset_Custom,
         'solar':Dataset_Custom,
         'fred':Dataset_Custom,
         'pems03':Dataset_Custom,
         'pems04':Dataset_Custom,
         'pems07':Dataset_Custom,
         'pems08':Dataset_Custom,
         'other':Dataset_Custom
        }      

def DataProvider(args):
    
    concat_dataset = []
    # config
    batch_size = args.batch_size
    drop_last = False
    dataset_list = args.dset_pretrain
    size = [args.context_points, 0, args.target_points]
    for dataset_name in dataset_list:
        factory = DSETS[dataset_name]
        dataset_kwargs={
                'root_path': 'data/',
                'data_path': dataset_name + ".csv",
                'features': args.features,
                'scale': True,
                'size': size,
                'use_time_features': False
                }
        dataset = factory(**dataset_kwargs,split='train')
        try:
            # print(f'{dataset_name} len: ', len(dataset))
            concat_dataset.append(dataset)
        except:
            pass

    concat_dataset = ConcatDataset(concat_dataset)

    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=drop_last,
        sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size))
    
    return concat_dataset, data_loader


class DataProviders():
    def __init__(self, args):
        # config
        self.batch_size = args.batch_size
        self.drop_last = False
        self.dataset_list = args.dset_pretrain
        self.num_workers = args.num_workers
        self.size = [args.context_points, 0, args.target_points]
        self.features = args.features
        self.half = args.finetune_percentage
        self.all = args.is_all
        self.one_channel=args.one_channel
        self.dset_path=str(args.dset_path)+'/'
        
        self.train = self.data_provider("train")
        if not self.all:
            self.test = self.data_provider("test")
            self.valid = self.data_provider("val")
        else:
            self.valid=None


    
    def concat_dataset(self,  split="train"):
        concat_dataset = []
        for dataset_name in self.dataset_list:
            if(dataset_name in DSETS):
                factory = DSETS[dataset_name]
            else:
                factory = DSETS['other']
            dataset_kwargs={
                    'root_path': self.dset_path,
                    'data_path': dataset_name + ".csv",
                    'features': self.features,
                    'scale': True,
                    'size': self.size,
                    'use_time_features': False,
                    'half':self.half,
                    'all':self.all,
                    'one_channel':self.one_channel
                    }
            dataset = factory(**dataset_kwargs,split=split)
            try:
                # print(f'{dataset_name} len: ', len(dataset))
                if len(dataset) > 0:
                    concat_dataset.append(dataset)
            except:
                pass
            
        concat_dataset = ConcatDataset(concat_dataset)

        return concat_dataset

    def data_provider(self, split, shuffle=False):
        concat_dataset = self.concat_dataset(split=split)
        if self.one_channel:
            data_loader = DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True)
        else:
            data_loader = DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=self.batch_size))
        return data_loader





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument('--dset_pretrain', type=list, default=['ETTh1','ETTm1','weather','exchange','illness','ETTh2','ETTm2'], help='dataset name')
    parser.add_argument('--context_points', type=int, default=96, help='sequence length')
    parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
    parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')

    args = parser.parse_args()

    dp = DataProviders(args)
    for batch in dp.train:
        a, b = batch
        print(a)
