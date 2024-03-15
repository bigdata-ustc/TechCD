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
        self.num_layers = 4
        self.discarding_layers = 2

        super(Net, self).__init__()

        # network structure
        self.entity = nn.Embedding(self.knowledge_dim + self.exer_n, self.stu_dim)
        self.stu = nn.Embedding(self.emb_num, self.stu_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.k_index = torch.LongTensor(list(range(self.knowledge_dim))).to(self.device)
        self.e_index = torch.LongTensor(list(range(self.knowledge_dim, self.knowledge_dim + self.exer_n))).to(self.device)
        self.entity_index = torch.LongTensor(list(range(self.knowledge_dim + self.exer_n))).to(self.device)

        self.similarity_gcn = self.create_gcn_layers(self.similarity_g, self.num_layers)
        self.prerequisite_gcn = self.create_gcn_layers(self.prerequisite_g, self.num_layers)
        self.exer_concept_gcn = self.create_gcn_layers(self.exer_concept_g, self.num_layers)

        self.disc = nn.Linear(self.knowledge_dim, 1)
        self.full_stu = nn.Linear(2*self.knowledge_dim,1)
        self.full_exer = nn.Linear(2*self.knowledge_dim,1)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def create_gcn_layers(self, graph, num_layers):
        gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            gcn_layers.append(GCN(graph, self.knowledge_dim, self.knowledge_dim, None))
        return gcn_layers

    def forward(self, stu_id, exer_id, kn_emb, input_knowedge_ids, history):
        '''
        :param:
        :return: FloatTensor, the probabilities of answering correctly
        '''

        entity_embeddings = [self.entity(self.entity_index)]
        for similarity_gcn_layer, prerequisite_gcn_layer, exer_concept_gcn_layer in zip(self.similarity_gcn, self.prerequisite_gcn, self.exer_concept_gcn):
            similarity_embeddings = similarity_gcn_layer(entity_embeddings[-1])
            prerequisite_embeddings = prerequisite_gcn_layer(entity_embeddings[-1])
            exer_concept_embeddings = exer_concept_gcn_layer(entity_embeddings[-1])
            entity_embeddings.append(similarity_embeddings + prerequisite_embeddings + exer_concept_embeddings)
        full_embeddings = torch.mean(torch.stack(entity_embeddings), dim=0)
        discarding_embeddings = torch.mean(torch.stack(entity_embeddings[self.discarding_layers+1:]), dim=0)

        concept_entity = full_embeddings[self.k_index]
        exer_entity = full_embeddings[self.e_index]
        stu_exe_entity = discarding_embeddings[self.e_index] # bottom discard

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
        input_x = torch.sigmoid(self.prednet_full1(input_x))
        input_x = torch.sigmoid(self.prednet_full2(input_x))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)



class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
