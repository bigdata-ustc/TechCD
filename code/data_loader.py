import json
import torch
import random

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, data=''):
        self.batch_size = 256
        self.ptr = 0
        self.data = []

        with open('../data/' + data + '/config.txt') as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = list(map(eval, i_f.readline().split(',')))
            self.knowledge_dim = int(self.knowledge_n)
        with open('../data/' + data + '/train.json', encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open('../data/' + data + '/history.json', encoding='utf8') as i_f:
            self.history_data = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, input_knowedge_ids, ys, history = [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowedge_ids.append(log['knowledge_code'][0]-1)
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            # print (self.embedding[log['exer_id']])
            if str(log['user_id']) not in self.history_data.keys():
                history.append([random.randint(1, self.exer_n + self.knowledge_n)])
            else:
                history.append(self.history_data[str(log['user_id'])])
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(input_knowedge_ids), torch.LongTensor(ys), history

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation', data=''):
        self.batch_size = 256
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        with open('../data/' + data + '/history.json', encoding='utf8') as i_f:
            self.history_data = json.load(i_f)
        data_file = '../data/' + data + '/test.json'
        config_file = '../data/' + data + '/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(self.knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, input_knowedge_ids, ys, history = [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowedge_ids.append(log['knowledge_code'][0] - 1)
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            # print (self.embedding[log['exer_id']])
            if str(log['user_id']) not in self.history_data.keys():
                history.append([random.randint(1, self.exer_n + self.knowledge_n)])
            else:
                history.append(self.history_data[str(log['user_id'])])
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(input_knowedge_ids), torch.LongTensor(ys), history

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class TestDataLoader(object):
    def __init__(self, d_type='test', data=''):
        self.batch_size = 256
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        with open('../data/' + data + '/history.json', encoding='utf8') as i_f:
            self.history_data = json.load(i_f)
        data_file = '../data/' + data + '/test.json'
        config_file = '../data/' + data + '/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            self.student_n, self.exer_n, self.knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(self.knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, input_knowedge_ids, ys, history = [], [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowedge_ids.append(log['knowledge_code'][0] - 1)
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            # print (self.embedding[log['exer_id']])
            if str(log['user_id']) not in self.history_data.keys():
                history.append([random.randint(1, self.exer_n, self.knowledge_n)])
            else:
                history.append(self.history_data[str(log['user_id'])])
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(input_knowedge_ids), torch.LongTensor(ys), history

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0