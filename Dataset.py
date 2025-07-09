import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

custom_edges = [(0, 5), (0, 9), (1, 4), (1, 5), (1, 9), (2, 3), (3, 5), (3, 9), (6, 7), (6, 5), (6, 9), (8, 5), (8, 9),
                (5, 9)]  

label =6
m = 4.5

def read_excel_data(file_path):
    df = pd.read_excel(file_path, header=None)
    nodes = df.iloc[1:11, 1:].values

    nodes = pd.to_numeric(nodes.flatten(), errors='coerce').reshape(-1, nodes.shape[1])

    return nodes



def create_graph_data(nodes, custom_edges, label):

    x = torch.tensor(nodes, dtype=torch.float)

    edge_index = set()
    edge_attr = []

    for edge in custom_edges:
        node1, node2 = edge

        edge_index.add((node1, node2))
        edge_index.add((node2, node1))

    edge_index = torch.tensor(list(edge_index), dtype=torch.long).t().contiguous()  # 转回list并排序
    print(edge_index.shape)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)


    y = torch.tensor([label], dtype=torch.long)


    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)



folder_path = "OUT"
output_folder = "DATA1"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

for excel_file in excel_files:
    file_path = os.path.join(folder_path, excel_file)

    nodes = read_excel_data(file_path)


    graph_data = create_graph_data(nodes, custom_edges, label)

    output_file = os.path.join(output_folder, f"{label}-{m}-{excel_file.split('.')[0]}.pt")
    torch.save(graph_data, output_file)

    print(f"图数据已保存：{output_file}")
    print(f"节点特征：\n{graph_data.x}")
    print(f"节点连接：\n{graph_data.edge_index}")
    print(f"边特征：\n{graph_data.edge_attr}")
    print(f"标签：\n{graph_data.y}")
