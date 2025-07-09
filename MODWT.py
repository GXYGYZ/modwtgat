import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from modwt import modwt


def load_excel_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    return files


def modwt_and_energy(signal, wavelet='db4', level=4):
    coeffs = modwt(signal, wavelet, level)
    coeffs_list = [coeffs[i] for i in reversed(range(coeffs.shape[0]))]
    energy = [np.sum(np.square(c)) for c in coeffs_list]
    return coeffs_list, energy


def extract_sampled_data(df, sample_interval):
    sampled_data = df.iloc[::int(sample_interval)]
    return sampled_data.reset_index(drop=True)


def process_sampled_data(df, wavelet='db4', level=4, minutes=1):
    energy_columns = ["Column Name", "Level 0 Energy", "Level 1 Energy", "Level 2 Energy",
                      "Level 3 Energy", "Level 4 Energy", "Sampling Time"]
    results = pd.DataFrame(columns=energy_columns)

    columns_to_process = df.columns[1:-3]

    for col in columns_to_process:
        if df[col].isnull().any():
            print(f"Skipping column {col} because it contains missing values")
            continue

        signal = df[col].values
        coeffs, energy = modwt_and_energy(signal, wavelet, level)
        row = [col] + energy + [minutes]
        results.loc[len(results)] = row
        plot_modwt_decomposition(signal, coeffs, col, level, wavelet)
    return results

# Plot MODWT decomposition
# def plot_modwt_decomposition(signal, coeffs, col_name, level, wavelet):
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.figure(figsize=(12, 10))
#
#  
#     plt.subplot(level + 2, 1, 1)
#     plt.plot(signal, color='#d19c20ff',linewidth=2.5)
#     # plt.title("Original Signal", fontsize=20)
#     # plt.tick_params(axis='both', labelsize=14)  
#   
#     # for spine in plt.gca().spines.values():
#     #     spine.set_linewidth(2) 
#
#     for i, coeff in enumerate(coeffs):
#         plt.subplot(level + 2, 1, i + 2)
#         plt.plot(coeff, color='green', linewidth=2.5)
#         # plt.title(f"Level {i}", fontsize=20)
#         # plt.tick_params(axis='both', labelsize=14)
#         #
#         # for spine in plt.gca().spines.values():
#         #     spine.set_linewidth(2) 
#
#     plt.tight_layout()
#     plt.show()

# Plot MODWT decomposition
import matplotlib.pyplot as plt


def plot_modwt_decomposition(signal, coeffs, col_name, level, wavelet):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(12, 10))


    plt.subplots_adjust(hspace=0)


    plt.subplot(level + 2, 1, 1)
    plt.plot(signal, color='#d19c20ff', linewidth=2.5)
    plt.gca().set_ylabel('')
    plt.gca().set_xticklabels([])

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)


    for i in range(level):
        plt.subplot(level + 2, 1, i + 2)
        plt.plot(coeffs[i], color='green', linewidth=2.5)
        plt.gca().set_ylabel('')
        plt.gca().set_xticklabels([])

        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)


    plt.subplot(level + 2, 1, level + 2)
    plt.plot(coeffs[-1], color='green', linewidth=2.5)
    plt.gca().set_ylabel('')

    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.show()

def process_file(file_path, output_folder, wavelet='db4', level=4, minutes=1):
    df = pd.read_excel(file_path)
    sample_interval = minutes * 2
    sampled_data = extract_sampled_data(df, sample_interval)
    results = process_sampled_data(sampled_data, wavelet, level, minutes)
    base_name = os.path.basename(file_path).replace('.xlsx', f'-{minutes}min.xlsx')
    output_file = os.path.join(output_folder, base_name)
    results.to_excel(output_file, index=False)
    print(f"Processing completed: {file_path}, results saved to {output_file}")


def process_folder(input_folder, output_folder, wavelet='db4', level=4, minutes=1):
    files = load_excel_files(input_folder)
    for file in files:
        file_path = os.path.join(input_folder, file)
        print(f"Processing file: {file}")
        process_file(file_path, output_folder, wavelet, level, minutes)


def process_all_folders(base_input_folder, base_output_folder, wavelet='db4', level=4, minutes=1):
    for root, dirs, files in os.walk(base_input_folder):
        for dir_name in dirs:
            input_folder = os.path.join(root, dir_name)
            output_folder = os.path.join(base_output_folder, dir_name)
            os.makedirs(output_folder, exist_ok=True)
            print(f"Processing folder: {input_folder}")
            process_folder(input_folder, output_folder, wavelet, level, minutes)


base_input_folder = 'IN'
base_output_folder = 'OUT'
minutes = 0.5


process_all_folders(base_input_folder, base_output_folder, wavelet='db4', level=4, minutes=minutes)
