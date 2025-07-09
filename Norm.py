import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler



def load_graph_data(folder_path):
    graph_data_list = []
    empty_graph_files = []
    sampling_intervals = []
    original_filenames = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt'):
            file_path = os.path.join(folder_path, file_name)
            graph_data = torch.load(file_path)


            if graph_data.x is None or graph_data.x.size(0) == 0:
                empty_graph_files.append(file_name)
                continue

            # 保存原始文件名
            original_filenames.append(file_name)


            last_feature = graph_data.x[:, -1].clone()  # 保存原始值
            sampling_intervals.append(last_feature.numpy())


            graph_data.x = graph_data.x[:, :-1]

            graph_data_list.append(graph_data)

    return graph_data_list, empty_graph_files, original_filenames, sampling_intervals



def compute_global_statistics(graph_data_list):
    all_nodes = []
    for graph_data in graph_data_list:
        if graph_data.x.size(0) > 0:
            all_nodes.append(graph_data.x.numpy())

    scaler = StandardScaler()
    if len(all_nodes) > 0:
        scaler.fit(np.vstack(all_nodes))
    return scaler



def normalize_graph_data(graph_data_list, scaler, sampling_intervals):
    for i, graph_data in enumerate(graph_data_list):
        
        normalized_features = scaler.transform(graph_data.x.numpy())


        final_features = np.hstack([
            normalized_features,
            sampling_intervals[i].reshape(-1, 1)
        ])

        graph_data.x = torch.tensor(final_features, dtype=torch.float32)
    return graph_data_list



def save_normalized_graphs(graph_data_list, output_folder, original_filenames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for graph_data, file_name in zip(graph_data_list, original_filenames):
        output_file = os.path.join(output_folder, file_name)
        torch.save(graph_data, output_file)
        print(f"已保存: {output_file}")



input_folder_path = 'DATA1'
output_folder_path = 'norm'


graph_data_list, empty_files, filenames, sampling_intervals = load_graph_data(input_folder_path)


if empty_files:
    print(f"空文件列表: {empty_files}")


scaler = compute_global_statistics(graph_data_list)


graph_data_normalized = normalize_graph_data(graph_data_list, scaler, sampling_intervals)


save_normalized_graphs(graph_data_normalized, output_folder_path, filenames)


sample_data = graph_data_normalized[0]
print("\n标准化验证:")
print(f"特征维度: {sample_data.x.shape}")
print(f"最后一个特征（采样间隔）示例值: {sample_data.x[0, -1].item()}")
print(f"标准化后特征均值: {sample_data.x[:, :-1].mean(dim=0)}")
print(f"标准化后特征标准差: {sample_data.x[:, :-1].std(dim=0)}")
