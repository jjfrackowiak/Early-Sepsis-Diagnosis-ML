import pandas as pd
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa


# Torch DataSet object
class SepsisData(Dataset):
    def __init__(self, X, y, seq_len, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.seq_len = seq_len
        self.beginning_index_dict = {}
        self.end_index_dict = {}
        self.X = X.sort_values(by=['File_Path', 'ICULOS']).reset_index(drop=True)
        self.calculate_indices()
        self.y = torch.tensor(self.X['HoursBeforeOnset'].values, dtype=torch.float32, device=self.device)
        self.X = torch.tensor(self.X[['SepsisLabelOrg']].values, dtype=torch.float32, device=self.device)
        
    def calculate_indices(self):
        i = 0
        start = 0
        end = 0
        length = len(self.X)
        for file_path, group in tqdm(self.X.groupby('File_Path')):
            length = len(self.X)
            start = group.index[0]
            while True:
                end = start + self.seq_len
                
                if end >= length:
                    end = length
                    
                if end != length:
                    start_patient_id = self.X.iloc[start]['File_Path']
                    end_patient_id = self.X.iloc[end-1]['File_Path']

                if start_patient_id == end_patient_id:
                    if end == length:
                        self.beginning_index_dict[i] = start
                        self.end_index_dict[i] = end
                        break
                    
                    self.beginning_index_dict[i] = start
                    self.end_index_dict[i] = end
                    start = start+1
                    i += 1
                else:
                    break
        
    def __len__(self):
        return len(self.end_index_dict.keys())

    def __getitem__(self, idx):
        start_slice = self.beginning_index_dict[idx]
        end_slice = self.end_index_dict[idx]
        return self.X[start_slice:end_slice], self.y[end_slice-1]

    
# Custom lightning dataloader
class SepsisDataLoader(pl.LightningDataModule):
    def __init__(self, train_dir, seq_len, X_scaler, X_imputer,
                 test_size, random_state, batch_size, training_size, num_workers, device, pin_memory=True):
        super().__init__()
        self.train_dir = train_dir
        self.seq_len = seq_len
        self.X_scaler = X_scaler
        self.X_imputer = X_imputer
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.sepsis_train = None
        self.sepsis_val = None
        self.sepsis_test = None
        self.train_patient_ids = None
        self.val_patient_ids = None
        self.test_patient_ids = None
        self.training_size = training_size

    def prepare_for_training(self):
        # Load the dataset
        self.train_df = pd.read_csv(self.train_dir)

        # Create X and y
        self.y = self.train_df[["SepsisLabel"]].values
        self.X = self.train_df[['SepsisLabelOrg','SepsisLabel', 'ICULOS', 'File_Path', 'HoursBeforeOnset']]
      
        # Separate patients with sepsis_label=1 and sepsis_label=0
        sepsis_label_1_ids = self.X[self.X['SepsisLabel'] == 1]['File_Path'].unique()
        sepsis_label_0_ids = self.X[self.X['SepsisLabel'] == 0]['File_Path'].unique()

        # Set training size 
        sepsis_label_1_ids = sepsis_label_1_ids[:round(len(sepsis_label_1_ids)*self.training_size)]
        sepsis_label_0_ids = sepsis_label_0_ids[:round(len(sepsis_label_0_ids)*self.training_size)]

        # Drop columns with missingness > 90% duplicated index
        self.X = self.X['SepsisLabelOrg','SepsisLabel', 'ICULOS', 'File_Path'] #, 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'], axis=1)
        
        # Split sepsis_label=1 patients
        train_sepsis_label_1_ids, temp_sepsis_label_1_ids = train_test_split(sepsis_label_1_ids, test_size=self.test_size, random_state=self.random_state)
        val_sepsis_label_1_ids, test_sepsis_label_1_ids = train_test_split(temp_sepsis_label_1_ids, test_size=0.5, random_state=42)

        # Split sepsis_label=0 patients
        train_sepsis_label_0_ids, temp_sepsis_label_0_ids = train_test_split(sepsis_label_0_ids, test_size=self.test_size, random_state=self.random_state)
        val_sepsis_label_0_ids, test_sepsis_label_0_ids = train_test_split(temp_sepsis_label_0_ids, test_size=0.5, random_state=42)

        # Combine the splits for both sepsis_label=1 and sepsis_label=0
        self.train_patient_ids = list(train_sepsis_label_1_ids) + list(train_sepsis_label_0_ids)
        self.val_patient_ids = list(val_sepsis_label_1_ids) + list(val_sepsis_label_0_ids)
        self.test_patient_ids = list(test_sepsis_label_1_ids) + list(test_sepsis_label_0_ids)
        print('train val test id nums', len(self.train_patient_ids), len(self.val_patient_ids),len(self.test_patient_ids))
        print('uniq filepaths, len filepaths',len(self.X['File_Path'].unique()), len(self.X['File_Path']))

        # Filter the data for training, validation, and test sets based on patient_id
        train_mask = self.X['File_Path'].isin(self.train_patient_ids)
        val_mask = self.X['File_Path'].isin(self.val_patient_ids)
        test_mask = self.X['File_Path'].isin(self.test_patient_ids)

        self.X_train = self.X[train_mask]
        self.X_val = self.X[val_mask]
        self.X_test = self.X[test_mask]
        print('flag0', self.X_train.shape, self.X_test.shape, self.X_val.shape)
        self.y_train = self.y[train_mask]
        self.y_val = self.y[val_mask]
        self.y_test = self.y[test_mask]

        # Omit the 'File_path' column
        self.columns_to_keep = [col for col in self.X_train.columns if col not in ['File_Path', 'SepsisLabel']]
        
        # Drop the label and add padding
        self.X_train = self.add_missing_rows(self.X_train, 'File_Path', fill_column='SepsisLabel')
        self.X_val = self.add_missing_rows(self.X_val, 'File_Path', fill_column='SepsisLabel')        
        self.X_test = self.add_missing_rows(self.X_test, 'File_Path', fill_column='SepsisLabel')
        
        # Create three instances of the custom dataset class
        self.sepsis_train = SepsisData(self.X_train,
                                        self.y_train,
                                        self.seq_len,
                                        device=self.device
                                              )
        self.sepsis_val = SepsisData(self.X_val,
                                     self.y_val,
                                     self.seq_len,
                                     device=self.device
                                             )
        self.sepsis_test = SepsisData(self.X_test,
                                       self.y_test,
                                       self.seq_len,
                                       device=self.device
                                             )
        self.X = None
        self.y = None
        self.X_imputer = None
        self.X_scaler = None

    def add_missing_rows(self, df, id_column, fill_column=None):
        # Group by the ID column
        grouped = df.groupby(id_column)
        
        # Function to add missing rows within each group
        def add_missing_rows_group(group):
            if len(group) < self.seq_len:
                num_rows_to_add = self.seq_len - len(group)
                if fill_column is not None:
                    first_row_value = group.iloc[0][fill_column]
                    missing_rows_data = {
                        col: [first_row_value if col == fill_column else 0 for _ in range(num_rows_to_add)] 
                        for col in df.columns
                    }
                else:
                    missing_rows_data = {col: [0] * num_rows_to_add for col in df.columns}
                missing_rows_data[id_column] = group[id_column].iloc[0]
                missing_rows = pd.DataFrame(missing_rows_data)
                return pd.concat([missing_rows, group], ignore_index=True)
            else:
                return group
        
        # Apply the function to each group and reset index
        result = grouped.apply(add_missing_rows_group).reset_index(drop=True)
        return result
            
    def train_dataloader(self):
        # Return the dataloader of train data
        return DataLoader(self.sepsis_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          drop_last=False,
                          num_workers=self.num_workers,
                          persistent_workers=True#,
                          #pin_memory=self.pin_memory
                          )

    def val_dataloader(self):
        # Return the dataloader of validation data
        return DataLoader(self.sepsis_val,
                          shuffle=False,
                          batch_size=self.batch_size,
                          drop_last=False,
                          num_workers=self.num_workers,
                          persistent_workers=True#,
                          #pin_memory=self.pin_memory
                          )

    def test_dataloader(self):
        # Return the dataloader of test data
        return DataLoader(self.sepsis_test,
                          shuffle=False,
                          batch_size=self.batch_size,
                          drop_last=False,
                          num_workers=self.num_workers,
                          persistent_workers=True#,
                          #pin_memory=True
                          )
