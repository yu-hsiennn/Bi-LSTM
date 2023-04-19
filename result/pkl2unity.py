import os, pickle
import numpy as np

file_name = "angle_choreo_all_1205_ca_01"
save_path = "unity_txt_file"

def read(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def print_list(data):
    print(data[30])
    print(data[30])

def Data_processing(data):
    world_positions_list = []
    for frame in data:
        temp_str = ""
        for idx, skeletons in enumerate(frame):
            if idx % 3 == 2:
                temp_str += (str((skeletons)) + ",")
            else:
                temp_str += (str(skeletons) + " ")
        temp_str = temp_str[:len(temp_str)-1]
        world_positions_list.append(temp_str)
    return world_positions_list

def list2txt(save_path, data_list):
    with open(save_path, 'w') as f:
        for coordinates in data_list:
            f.write(coordinates)
            f.write("\n")

GT_data = read(f"{file_name}_ori.pkl")
model_data = read(f"{file_name}.pkl")

print("---------GT data---------")
print_list(GT_data)
print("---------model data----------")
print_list(model_data)

list2txt(f"{save_path}/{file_name}_ori.txt", Data_processing(GT_data))
list2txt(f"{save_path}/{file_name}.txt", Data_processing(model_data))