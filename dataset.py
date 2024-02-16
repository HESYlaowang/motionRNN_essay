import os.path
import os
import pickle
import numpy as np
import gzip

key_file = {
    'train_data' : 'train_data_1000.gz',
    'train_label' : 'train_label_1000.gz',
    'valid_data' : 'valid_data_100.gz',
    'valid_label' : 'valid_label_100.gz',
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/dataset.pkl"

train_num = 1000
test_num = 100
data_dim = (4, 800, 1)
data_size = 3200

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_data(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, data_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_data'] =  _load_data(key_file['train_data'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['valid_data'] = _load_data(key_file['valid_data'])
    dataset['valid_label'] = _load_label(key_file['valid_label'])

    return dataset

def init_dataset():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_dataset(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_dataset()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_data', 'valid_data'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['valid_label'] = _change_one_hot_label(dataset['vaild_label'])

    if not flatten:
         for key in ('train_data', 'valid_data'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_data'], dataset['train_label']), (dataset['valid_data'], dataset['valid_label'])


if __name__ == '__main__':
    init_dataset()