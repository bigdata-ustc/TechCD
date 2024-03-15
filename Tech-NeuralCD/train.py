import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from build_graph import build_graph

# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')

data = 'CrossSchool' # dataset includes ASSIST, Junyi, CrossSchool

with open('../data/' + data + '/config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

local_map = {
    'prerequisite_g': build_graph('prerequisite', knowledge_n + exer_n, data),
    'similarity_g': build_graph('similarity', knowledge_n + exer_n, data),
    'exer_concept_g': build_graph('exer_concept', knowledge_n + exer_n, data)
}
epoch_n = 20
learning_rate = 0.001

def train():
    data_loader = TrainDataLoader(data)
    net = Net(student_n, exer_n, knowledge_n, local_map, device)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, input_knowedge_ids, labels, history = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, input_knowedge_ids, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), input_knowedge_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, input_knowedge_ids, history)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+0.0001), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse, auc = validate(net, epoch)
        save_snapshot(net, '../model/' + data + '/model_epoch' + str(epoch + 1))


def validate(net, epoch):
    data_loader = ValTestDataLoader('validation', data)
    print('validating model...')
    data_loader.reset()
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all, binary_pre = [], [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, input_knowedge_ids, labels, history = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, input_knowedge_ids, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), input_knowedge_ids.to(device), labels.to(device)
        # print(input_stu_ids.shape)
        # print(exer_emb.shape)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, input_knowedge_ids, history)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
            if output[i] >= 0.5:
                binary_pre.append(int(1))
            else:
                binary_pre.append(int(0))
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    mae = np.mean(np.sqrt((label_all - pred_all) ** 2))
    binary_pre = np.array(binary_pre)
    f1 = f1_score(label_all, binary_pre)
    precision = precision_score(label_all, binary_pre)
    recall = recall_score(label_all, binary_pre)
    print('domain: %s, accuracy= %f, rmse= %f, auc= %f, f1_score=%f, precision=%f, recall=%f, mae=%f' % (data, accuracy, rmse, auc, f1, precision, recall, mae))
    with open('../result/' + data + '/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, f1_score=%f, precision=%f, recall=%f, mae=%f\n' % (epoch+1, accuracy, rmse, auc, f1, precision, recall,mae))
    return rmse, auc

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    # global student_n, exer_n, knowledge_n, device
    train()
