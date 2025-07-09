import os
import torch
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import math
import csv
from collections import Counter
from torch_geometric.nn import GNNExplainer
import matplotlib.pyplot as plt


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


folder_path = 'norm'


class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_units, num_heads=4, dropout_rate=0.3):
        super(GAT, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = GATConv(in_channels, hidden_units, heads=num_heads, concat=True, add_self_loops=False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_units * num_heads)
        self.conv2 = GATConv(hidden_units * num_heads, hidden_units, heads=4, concat=True, add_self_loops=False)
        self.bn2 = torch.nn.BatchNorm1d(hidden_units * 4)
        self.conv3 = GATConv(hidden_units * 4, hidden_units, heads=1, concat=False, add_self_loops=False)
        self.bn3 = torch.nn.BatchNorm1d(hidden_units)
        self.fc = torch.nn.Linear(hidden_units, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.fc(x)



def load_graphs_from_folder(folder_path):
    graph_data_list = []
    labels = []
    sampling_intervals = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)
            data = torch.load(file_path)


            interval = round(data.x[0, -1].item(), 4)
            sampling_intervals.append(interval)

            graph_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if 'edge_attr' in data else None,
                y=data.y
            )
            graph_data_list.append(graph_data)
            labels.append(data.y.item())

    return graph_data_list, labels, sampling_intervals



def create_strata(labels, intervals):
    combined_strata = [f"{label}_{interval}" for label, interval in zip(labels, intervals)]


    strata_counts = Counter(combined_strata)
    final_strata = []
    for s in combined_strata:
        if strata_counts[s] >= 2:
            final_strata.append(s)

    return final_strata



def train(model, train_loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()


        noise = torch.randn_like(data.x) * 0.0001
        data.x = data.x + noise

        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    return total_loss / len(train_loader), current_lr



def test(model, test_loader, criterion):
    model.eval()
    num_classes = 6
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    total_loss = 0
    total_samples = 0
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    with torch.no_grad():
        for data in test_loader:
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y - 1)
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

            _, predicted = torch.max(output, dim=1)
            target = data.y - 1
            for i in range(num_classes):
                correct[i] += ((predicted == i) & (target == i)).sum().item()
                total[i] += (target == i).sum().item()
                true_positives[i] += ((predicted == i) & (target == i)).sum().item()
                false_positives[i] += ((predicted == i) & (target != i)).sum().item()
                false_negatives[i] += ((predicted != i) & (target == i)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = correct / total
    average_loss = total_loss / total_samples
    overall_accuracy = correct.sum() / total.sum()

    metrics = {
        "Precision": precision.tolist(),
        "Recall": recall.tolist(),
        "F1-Score": f1_score.tolist(),
        "Accuracy": accuracy.tolist(),
    }
    return overall_accuracy, average_loss, metrics



def get_correct_predictions(model, test_loader, correct_class):
    per_class_correct = {c: [] for c in range(1, 7)}
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            output = model(batch.x, batch.edge_index, batch.batch)
            _, predicted = torch.max(output, dim=1)
            predicted_labels = predicted + 1  # 预测标签加1后与原始标签比较


            graph_list = batch.to_data_list()

            for i, graph in enumerate(graph_list):
                true_label = graph.y.item()
                pred_label = predicted_labels[i].item()

                if true_label == pred_label and true_label == correct_class:
                    per_class_correct[true_label].append(graph)

    return per_class_correct[correct_class]


# 解释并平均模型预测
def explain_and_average(model, graphs, correct_class):
    if not graphs:
        print(f"No correct predictions for class {correct_class}")
        return None, None

    node_feat_masks = []
    edge_masks = []
    explainer = GNNExplainer(model, epochs=100, return_type='prob')

    for graph in graphs:
        node_feat_mask, edge_mask = explainer.explain_graph(graph.x, graph.edge_index)
        node_feat_masks.append(node_feat_mask)
        edge_masks.append(edge_mask)


    avg_node_feat_mask = torch.mean(torch.stack(node_feat_masks), dim=0)
    avg_edge_mask = torch.mean(torch.stack(edge_masks), dim=0)

    return avg_node_feat_mask, avg_edge_mask


def analyze_node_features(node_feat_mask):

    feature_names = [
        'Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Intervals'
    ]

    print("\n=== 节点特征重要性分析 ===")


    if node_feat_mask.dim() == 2:

        feature_importances = node_feat_mask.mean(dim=0)
    else:

        feature_importances = node_feat_mask


    print("\n所有特征及其重要性:")
    for i, name in enumerate(feature_names):
        imp = feature_importances[i].item()
        print(f"{name}: {imp:.4f}")

    return feature_names, feature_importances


def visualize_feature_importance(feature_names, importances, class_label):

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances.tolist(), color='skyblue')
    plt.xlabel('特征')
    plt.ylabel('平均重要性')
    plt.title(f'类别 {class_label} 的特征重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()


    img_path = f"feature_importance_class_{class_label}.png"
    plt.savefig(img_path)
    plt.close()
    print(f"已保存特征重要性图到: {img_path}")

def analyze_important_edges(edge_mask, edge_index):

    part_mapping = {
        0: "前额",
        1: "右大腿",
        2: "右手背",
        3: "右手上臂",
        4: "右小腿前",
        5: "左背",
        6: "左手下臂",
        7: "左手腕",
        8: "左小腿后",
        9: "左胸",
    }

    print("\n=== 重要边分析 ===")

    print("\n所有边及其权重:")
    for i in range(len(edge_mask)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        weight = edge_mask[i].item()
        u_part = part_mapping.get(u, f"未知({u})")
        v_part = part_mapping.get(v, f"未知({v})")
        print(f"边 {u_part} -> {v_part}: {weight:.4f}")


    top_indices = torch.argsort(edge_mask, descending=True)[:5]
    print("\n前5个最重要的边:")
    for idx in top_indices:
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        weight = edge_mask[idx].item()
        u_part = part_mapping.get(u, f"未知({u})")
        v_part = part_mapping.get(v, f"未知({v})")
        print(f"边 {u_part} -> {v_part}: {weight:.4f}")


    bottom_indices = torch.argsort(edge_mask)[:5]
    print("\n前5个最不重要的边:")
    for idx in bottom_indices:
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        weight = edge_mask[idx].item()
        u_part = part_mapping.get(u, f"未知({u})")
        v_part = part_mapping.get(v, f"未知({v})")
        print(f"边 {u_part} -> {v_part}: {weight:.4f}")



def stratified_cross_validation():

    graph_data_list, labels, sampling_intervals = load_graphs_from_folder(folder_path)
    combined_strata = create_strata(labels, sampling_intervals)


    n_strata = len(set(combined_strata))
    min_test_ratio = max(0.1, n_strata / len(graph_data_list) + 0.05)
    test_ratio = min(0.3, min_test_ratio)  # 防止测试集过大

    remaining_data, test_data, remaining_strata, _ = train_test_split(
        graph_data_list, combined_strata,
        test_size=test_ratio,
        stratify=combined_strata,
        random_state=seed
    )


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_fold_results = []


    all_explanations = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(remaining_data, remaining_strata)):
        print(f"\n=== Fold {fold + 1}/5 ===")


        fold_train_data = [remaining_data[i] for i in train_idx]
        fold_val_data = [remaining_data[i] for i in val_idx]


        train_loader = DataLoader(fold_train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(fold_val_data, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


        model = GAT(
            in_channels=fold_train_data[0].x.shape[1],
            hidden_units=32,
            num_classes=6
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


        def lr_lambda(epoch):
            n = 90
            min_lr = 0.0001
            max_lr = 0.0009
            epochs = 200
            if epoch < n:
                return 1.0
            else:
                return (min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(epoch / epochs * math.pi))) / 0.001

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        criterion = torch.nn.CrossEntropyLoss()


        best_val_acc = 0
        for epoch in range(200):
            train_loss, current_lr = train(model, train_loader, optimizer, criterion, scheduler)


            val_acc, val_loss, _ = test(model, val_loader, criterion)


            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
                print(f'Fold {fold} Epoch {epoch}: Best Val Acc {val_acc:.4f}')


            if epoch % 20 == 0:
                print(
                    f'Fold {fold} Epoch {epoch} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        test_acc, test_loss, test_metrics = test(model, test_loader, criterion)
        print(f'Fold {fold} Test Accuracy: {test_acc:.4f}')


        all_fold_results.append({
            'fold': fold,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'precision': test_metrics["Precision"],
            'recall': test_metrics["Recall"],
            'f1': test_metrics["F1-Score"]
        })


        fold_explanations = {}
        for class_label in range(1, 7):
            print(f"\n为类别 {class_label} 解释模型...")

            correct_graphs = get_correct_predictions(model, test_loader, class_label)
            print(f"找到 {len(correct_graphs)} 个类别 {class_label} 的正确预测图")

            if correct_graphs:

                avg_node_feat_mask, avg_edge_mask = explain_and_average(model, correct_graphs, class_label)


                edge_index = correct_graphs[0].edge_index


                print(f"\n类别 {class_label} 的重要边分析:")
                analyze_important_edges(avg_edge_mask, edge_index)
                feature_names, feature_importances = analyze_node_features(avg_node_feat_mask)
                visualize_feature_importance(feature_names, feature_importances, class_label)

                fold_explanations[class_label] = {
                    'avg_node_mask': avg_node_feat_mask,
                    'avg_edge_mask': avg_edge_mask
                }
            else:
                print(f"类别 {class_label} 没有找到正确预测的图")

        all_explanations[fold] = fold_explanations


    with open('crossval_results.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'test_acc', 'test_loss', 'precision', 'recall', 'f1'])
        writer.writeheader()
        writer.writerows(all_fold_results)


    avg_acc = np.mean([res['test_acc'] for res in all_fold_results])
    print(f"\nAverage Test Accuracy: {avg_acc:.4f}")
    avg_precision = np.mean([np.mean(res['precision']) for res in all_fold_results])
    avg_recall = np.mean([np.mean(res['recall']) for res in all_fold_results])
    avg_f1 = np.mean([np.mean(res['f1']) for res in all_fold_results])


    print("\n=== Final Evaluation Metrics ===")
    print(f"Average Test Accuracy: {avg_acc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")


    torch.save(all_explanations, 'all_explanations.pt')
    print("\n模型解释结果已保存到 'all_explanations.pt'")



if __name__ == "__main__":
    stratified_cross_validation()
