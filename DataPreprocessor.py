import pathlib
import pandas as pd
import numpy as np
import pyprind

class DataPreprocessor():
    def __init__(self, session):
        self.session = session

        self.dir_path = pathlib.Path('data')
        col_names = ['Tag', 'Code',
                     'Group', 'Subgroup',
                     'Prog', 'Gender',
                     'Age', 'Language',
                     'Nationality', 'Domain']
        self.ds = pd.read_csv(self.dir_path.joinpath('dataset.csv'), names=col_names)
        self.ds = self.ds.dropna()
        self.ds = self.ds[self.ds['Group'] == session.lower()]

    def preprocess_labels(self, epsilon_t, epsilon_d):

        label_mapping = {label: idx for idx, label in enumerate(np.unique(self.ds['Tag']))}
        n = self.ds['Tag'].to_numpy().shape[0]
        meet_mat = np.zeros((n,n))

        self.label_mapping = {label: idx for idx, label
                              in enumerate(np.unique(self.ds['Tag']))}

        session_path = self.dir_path.joinpath('Session' + self.session.lower())
        file_list = sorted([str(path) for path in session_path.glob('*.csv')])

        pos_ds_list = [pd.read_csv(file, sep=';') for file in file_list]
        pos_ds = pd.concat(pos_ds_list)

        pos_ds['TimeStamp'] = pos_ds['TimeStamp'].map(lambda x:
                                                      x.replace('/', '-'))
        pos_ds['TimeStamp'] = pd.to_datetime(pos_ds['TimeStamp']) - pd.to_datetime('01-01-2000')
        pos_ds['TimeStamp'] = pos_ds['TimeStamp'].dt.total_seconds()
        pos_ds = pos_ds.sort_values(['TimeStamp'], ignore_index=True)
        bar = pyprind.ProgBar(pos_ds.shape[0])
        for i in range(pos_ds.shape[0]-1):
            bar.update()
            for j in range(i+1, pos_ds.shape[0]-1):
                tag_index_1  = label_mapping[pos_ds['TagId'][i]]
                tag_index_2 = label_mapping[pos_ds['TagId'][j]]
                if tag_index_1 != tag_index_2:
                    meet_time = pos_ds['TimeStamp'][j] - pos_ds['TimeStamp'][i]
                    if (meet_time < epsilon_t):
                        x = pos_ds['X'][i] - pos_ds['X'][j]
                        y = pos_ds['Y'][i] - pos_ds['Y'][j]
                        distance = np.sqrt(x**2 + y**2)
                        if distance < epsilon_d:
                            meet_mat[tag_index_1, tag_index_2] = 1
                            meet_mat[tag_index_2, tag_index_1] = 1
                    else:
                        break
        return meet_mat
