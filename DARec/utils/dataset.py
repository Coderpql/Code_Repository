import torch
from torch.utils.data import Dataset
import utils.data_process as data_process


# AutoRec数据
class AutoRecDataset(Dataset):
    def __init__(self, args, domain, mode='train', is_process=False, is_filter=False):
        super(AutoRecDataset, self).__init__()
        if is_filter:
            # 按照目标域和辅助域组合进行数据筛选
            self.combinations = args['combinations']
            data_process.process_CD_data(args['path_data'], self.combinations)
        self.path_data = args['path_data'] + domain + '_.csv'
        self.path = args['path']
        self.ratio = args['ratio']
        self.mode = mode
        self.is_process = is_process
        self.domain = domain
        self.data = data_process.get_data_autoRec(self.path_data, self.path, self.ratio,
                                                  self.domain, self.is_process, self.mode)
        self.data = torch.FloatTensor(self.data)

    def __getitem__(self, item):
        return self.data[item], self.data[item]

    def __len__(self):
        return self.data.shape[0]


# UDARec数据
class UDARecDataset(Dataset):
    def __init__(self, args, target, source, mode='train', is_process=False):
        super(UDARecDataset, self).__init__()
        self.path_data = args['DARec']['parameter']['path_data']
        self.path = args['DARec']['parameter']['path']
        self.ratio = args['DARec']['parameter']['ratio']
        self.is_process = is_process
        self.mode = mode
        self.data_T, self.data_S = data_process.get_data_UDARec(self.path_data, self.path, target,
                                                                source, self.ratio, self.mode, self.is_process)

        self.data_T = torch.FloatTensor(self.data_T)
        self.data_S = torch.FloatTensor(self.data_S)

    def __getitem__(self, item):
        return self.data_T[:, :-1][item], self.data_T[:, :-1][item], self.data_T[:, -1][item], \
               self.data_S[:, :-1][item], self.data_S[:, :-1][item], self.data_S[:, -1][item]

    def __len__(self):
        return self.data_T.shape[0]
