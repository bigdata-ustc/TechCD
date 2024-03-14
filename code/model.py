import torch
import torch.nn as nn
from gnn import GCN
import torch.nn.functional as F

class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, local_map, device):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 216
        self.device = device
        self.similarity_g = local_map['similarity_g'].to(self.device)
        self.prerequisite_g = local_map['prerequisite_g'].to(self.device)
        self.exer_concept_g = local_map['exer_concept_g'].to(self.device)

        super(Net, self).__init__()

        # network structure
        self.entity = nn.Embedding(self.knowledge_dim + self.exer_n, self.stu_dim)
        self.stu = nn.Embedding(self.emb_num, self.stu_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.k_index = torch.LongTensor(list(range(self.knowledge_dim))).to(self.device)
        self.e_index = torch.LongTensor(list(range(self.knowledge_dim, self.knowledge_dim + self.exer_n))).to(self.device)
        self.entity_index = torch.LongTensor(list(range(self.knowledge_dim + self.exer_n))).to(self.device)

        self.similarity_gcn1 = GCN(self.similarity_g, self.knowledge_dim, self.knowledge_dim, None)
        self.prerequisite_gcn1 = GCN(self.prerequisite_g, self.knowledge_dim, self.knowledge_dim, None)
        self.exer_concept_gcn1 = GCN(self.exer_concept_g, self.knowledge_dim, self.knowledge_dim, None)

        self.similarity_gcn2 = GCN(self.similarity_g, self.knowledge_dim, self.knowledge_dim, None)
        self.prerequisite_gcn2 = GCN(self.prerequisite_g, self.knowledge_dim, self.knowledge_dim, None)
        self.exer_concept_gcn2 = GCN(self.exer_concept_g, self.knowledge_dim, self.knowledge_dim, None)

        self.similarity_gcn3 = GCN(self.similarity_g, self.knowledge_dim, self.knowledge_dim, None)
        self.prerequisite_gcn3 = GCN(self.prerequisite_g, self.knowledge_dim, self.knowledge_dim, None)
        self.exer_concept_gcn3 = GCN(self.exer_concept_g, self.knowledge_dim, self.knowledge_dim, None)

        self.similarity_gcn4 = GCN(self.similarity_g, self.knowledge_dim, self.knowledge_dim, None)
        self.prerequisite_gcn4 = GCN(self.prerequisite_g, self.knowledge_dim, self.knowledge_dim, None)
        self.exer_concept_gcn4 = GCN(self.exer_concept_g, self.knowledge_dim, self.knowledge_dim, None)

        self.disc = nn.Linear(self.knowledge_dim, 1)
        self.full_stu = nn.Linear(2*self.knowledge_dim,1)
        self.full_exer = nn.Linear(2*self.knowledge_dim,1)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = nn.Linear(self.prednet_len1, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb, input_knowedge_ids, history):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        # print (history)
        entity_emb = self.entity(self.entity_index)

        entity_emb_sim = self.similarity_gcn1(entity_emb)
        entity_emb_pre = self.prerequisite_gcn1(entity_emb)
        entity_emb_exe_con = self.exer_concept_gcn1(entity_emb)
        entity_emb1 = entity_emb_sim + entity_emb_pre + entity_emb_exe_con

        entity_emb_sim = self.similarity_gcn2(entity_emb1)
        entity_emb_pre = self.prerequisite_gcn2(entity_emb1)
        entity_emb_exe_con = self.exer_concept_gcn2(entity_emb1)
        entity_emb2 = entity_emb_sim + entity_emb_pre + entity_emb_exe_con

        entity_emb_sim = self.similarity_gcn3(entity_emb2)
        entity_emb_pre = self.prerequisite_gcn3(entity_emb2)
        entity_emb_exe_con = self.exer_concept_gcn3(entity_emb2)
        entity_emb3 = entity_emb_sim + entity_emb_pre + entity_emb_exe_con

        entity_emb_sim = self.similarity_gcn4(entity_emb3)
        entity_emb_pre = self.prerequisite_gcn4(entity_emb3)
        entity_emb_exe_con = self.exer_concept_gcn4(entity_emb3)
        entity_emb4 = entity_emb_sim + entity_emb_pre + entity_emb_exe_con

        # print ()
        # print (entity_emb1.shape)
        stu_entity = (entity_emb3 + entity_emb2)/2
        exer_entity = (entity_emb + entity_emb1 + entity_emb2 + entity_emb3)/4
        concept_entity = exer_entity[self.k_index]
        exer_entity = exer_entity[self.e_index]
        stu_exe_entity = stu_entity[self.e_index] # bottom discard

        stu_emb = torch.zeros([stu_id.shape[0], self.knowledge_dim]).to(self.device)
        for s in range(len(history)):
            s_id = torch.LongTensor(history[s]).to(self.device)
            s_emb = stu_exe_entity[s_id]
            stu_emb[s] = torch.mean(s_emb, dim=0)
        exer_emb = exer_entity[exer_id]

        stu_concept = concept_entity[self.k_index]


        batch_stu_vector = stu_emb.repeat(1, stu_emb.shape[1]).reshape(stu_emb.shape[0], stu_emb.shape[1], stu_emb.shape[1])

        kn_vector = stu_concept.repeat(stu_emb.shape[0], 1).reshape(stu_emb.shape[0], stu_concept.shape[0], stu_concept.shape[1])

        proficiency = torch.sigmoid(self.full_stu(torch.cat((batch_stu_vector, kn_vector), dim=2))).squeeze(2)
        batch_exer_vector = exer_emb.repeat(1, exer_emb.shape[1]).reshape(exer_emb.shape[0], exer_emb.shape[1], exer_emb.shape[1])
        k_difficulty = torch.sigmoid(self.full_exer(torch.cat((batch_exer_vector, kn_vector), dim=2))).squeeze(2)
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))*10

        # prednet
        input_x = e_discrimination * (proficiency - k_difficulty) * kn_emb
        output = torch.sigmoid(self.prednet_full1(input_x))
        output = torch.sigmoid(self.prednet_full2(input_x))
        output = torch.sigmoid(self.prednet_full3(input_x))

        # print (output)
        # exit(0)

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
