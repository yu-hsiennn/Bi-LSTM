import numpy as np
import pandas as pd
import os, re, random, collections
import torch
import pickle

def get_train_data(dataset, dir, ca, inp_len, out_len, randomInput):
    """
    Loads and processes training data from the specified dataset directory.

    Args:
        dataset (str): Path to the root dataset directory.
        dir (str): Subdirectory within the dataset directory.
        ca (str): class of dataset. 
        inp_len (int): Length of the input sequence.
        out_len (int): Length of the output sequence.
        randomInput (bool): Whether to use random input sequences.

    Returns:
        dict: A dictionary containing processed training data with input (x) and output (y) sequences.

    Raises:
        FileNotFoundError: If the specified dataset directory or its subdirectory is not found.
    """
    files_path = os.path.join(dataset, dir)
    if not os.path.exists(files_path):
        raise FileNotFoundError(f"Can't Find Dataset: {files_path}")
    
    files = sorted(os.listdir(files_path))
    
    traindatas = []
    ca = ca.split("_")
    total_frame = 0
    
    choose_files = []
    for file in files:
        # file name: angle_choreo_21joints_3_ca_01.pkl
        if re.split("_|\.", file)[-2] in ca :
            choose_files.append(file)
            with open(os.path.join(files_path, file), 'rb') as f:
                data = pickle.load(f)
            
            # Split the data into input and output sequences
            for i in range(0, len(data)-(inp_len+out_len), 5):
                if randomInput:
                    rand = random.randint(0, int(inp_len/2))
                else:
                    rand = int(inp_len/2)

                input_data = np.array(data[i:i+(rand*2)+out_len])
                output_data =  np.array(data[i:i+(rand*2)+out_len])
                # Set specific values in the input data to 1
                input_data[rand : rand+out_len] = 1
                traindatas.append((input_data, output_data))
                total_frame += (inp_len + out_len)
    print(f"total frame: {total_frame}")
    
    # save training file list
    with open("datas_name.txt",'w') as f:
        f.write('\n'.join(choose_files))

    data = collections.defaultdict(list)
    for i, (x,y) in enumerate(traindatas):
        data["x"].append(torch.tensor(x.astype("float32")))        
        data["y"].append(torch.tensor(y.astype("float32")))

    data["x"] = torch.nn.utils.rnn.pad_sequence(data["x"], batch_first=True)
    data["y"] = torch.nn.utils.rnn.pad_sequence(data["y"], batch_first=True)
    
    return data

def get_part_train_data(dataset, dir, ca, part, inp_len, out_len, randomInput, joint_def):
    """
    Loads and processes training part data from the specified dataset directory.

    Args:
        dataset (str): Path to the root dataset directory.
        dir (str): Subdirectory within the dataset directory.
        ca (str): class of dataset. 
        inp_len (int): Length of the input sequence.
        out_len (int): Length of the output sequence.
        randomInput (bool): Whether to use random input sequences.
        joint_def (class): class of model_joints

    Returns:
        dict: A dictionary containing processed training data with input (x) and output (y) sequences.

    Raises:
        FileNotFoundError: If the specified dataset directory or its subdirectory is not found.
    """
    files_path = os.path.join(dataset, dir)
    if not os.path.exists(files_path):
        raise FileNotFoundError(f"Can't Find Dataset: {files_path}")
    
    files = sorted(os.listdir(files_path))
    
    traindatas = []
    ca = ca.split("_")
    for file in files:
        if re.split("_|\.", file)[-2] in ca :
            with open(os.path.join(files_path, file), 'rb') as f:
                data = pickle.load(f)
            data = joint_def.cat_numpy(part, data)
            for i in range(0, len(data)-(inp_len+out_len), 5):
                if randomInput:
                    rand = random.randint(0, int(inp_len/2))
                else:
                    rand = int(inp_len/2)

                input_data = np.array(data[i:i+(rand*2)+out_len])
                output_data =  np.array(data[i: i+(rand*2)+out_len])
                input_data[rand: rand+out_len] = 1
                traindatas.append((input_data, output_data))

    data = collections.defaultdict(list)
    for i, (x,y) in enumerate(traindatas):
        data["x"].append(torch.tensor(x.astype("float32")))        
        data["y"].append(torch.tensor(y.astype("float32")))

    data["x"] = torch.nn.utils.rnn.pad_sequence(data["x"], batch_first=True)
    data["y"] = torch.nn.utils.rnn.pad_sequence(data["y"], batch_first=True)

    return data

def load_Tpos(Tpos_path):
    with open(Tpos_path, 'rb') as fpick:
        TPose = pickle.load(fpick)[0]
    return TPose

def load_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as fpick:
        data = pickle.load(fpick)
    return data

def load_xlsx_file(xlsx_path):
    """
    Load motion data files specified in an Excel file.

    Args:
    xlsx_path (str): Path to the Excel file containing data file paths.

    Returns:
    list: A list of motion data(coordinates) from the xlsx files.

    Note: This function reads data paths from an Excel file and loads the corresponding data files
    using the load_pkl_file function. It returns a list of loaded data.
    """
    df = pd.read_excel(xlsx_path)
    files = df['data_path'].tolist()
    files_data = [load_pkl_file(file) for file in files]
    return files_data

def save_pkl_file(pkl_path, data):
    with open(pkl_path, 'wb') as fpick:
        pickle.dump(data, fpick)
    print(f"pickle file was saved at :{pkl_path}")