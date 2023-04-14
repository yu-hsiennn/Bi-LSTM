import os, pickle

file_name = "S09_03_01_0926_V3_Human36M_train_angle_01_2010"
pkl_folder = "pkl_file"
saved_path = "unity_txt"

def read(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def print_list(data):
    print(data[-11])
    # print(len(data))

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

def list2txt(save_path, data_list, batch=False):
    with open(save_path, 'w') as f:
        for coordinates in data_list:
            f.write(coordinates)
            f.write("\n")
    name = save_path.split("\\") if batch else [saved_path, file_name]
    print(f"finished! The txt file was saved at :{name[0]}, file name: {name[1]}")

def batch_processing(source):
    all_pkl_files = os.listdir(source)
    for file in all_pkl_files:
        new_file_name = file[:-3] + "txt"
        list2txt(os.path.join(saved_path, new_file_name), Data_processing(read(os.path.join(source, file))), batch=True)
    print("done!")

# one file
# model_data = read(f"{file_name}.pkl")
# list2txt(f"{os.path.join(saved_path, file_name)}.txt", Data_processing(model_data))

# batch processing
# batch_processing(pkl_folder)

