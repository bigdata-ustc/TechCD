# -*- coding: utf-8 -*-

import dgl
import torch
import networkx as nx

import matplotlib.pyplot as plt


def build_graph(type, node, data):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    if type == 'prerequisite':
        with open('../data/' + data + '/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        edge_list = list(set(edge_list))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'similarity':
        with open('../data/' + data + '/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        edge_list = list(set(edge_list))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        return g
    elif type == 'exer_concept':
        with open('../data/' + data + '/Exer_Concept.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        edge_list = list(set(edge_list))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        # print (g)
        return g
