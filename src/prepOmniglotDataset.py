import os
import numpy as np
import random
from google_drive_downloader import GoogleDriveDownloader as gdd

# dataset path
dataset_path = 'datasets/omniglot_resized'

# download the Omniglot dataset
if not os.path.isdir(dataset_path):
    gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                        dest_path=dataset_path,
                                        unzip=True)

assert os.path.isdir(dataset_path)

# PrepData() class
class PrepData(object):
    def __init__(self, num_classes, num_meta_test_classes, config={}):
        """
        Args:
          num_classes: Number of classes for classification (K-way)
          num_meta_test_classes: Number of classes for classification (K-way) at meta-test time
        """
        self.num_classes = num_classes
        self.num_meta_test_classes = num_meta_test_classes

        data_folder = config.get('data_folder', dataset_path)
        self.img_size = config.get('img_size', (28, 28))

        character_folders = [os.path.join(data_folder, family, character)
                   for family in os.listdir(data_folder)
                   if os.path.isdir(os.path.join(data_folder, family))
                   for character in os.listdir(os.path.join(data_folder, family))
                   if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(123)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metatrain_character_labels = [np.random.randint(low=0, high=self.num_classes, size=1)[0] for i in range(len(self.metatrain_character_folders))]
        self.metaval_character_folders = character_folders[
          num_train:num_train + num_val]
        self.metaval_character_labels = [np.random.randint(low=0, high=self.num_classes, size=1)[0] for i in range(len(self.metaval_character_folders))]
        self.metatest_character_folders = character_folders[
          num_train + num_val:]
        self.metatest_character_labels = [np.random.randint(low=0, high=self.num_meta_test_classes, size=1)[0] for i in range(len(self.metatest_character_folders))]

        ## dataset info
        # training dataset
        self.metatrain_dataset = {}
        for i, cls_ in enumerate(self.metatrain_character_labels):
            if cls_ not in self.metatrain_dataset.keys():
                self.metatrain_dataset[cls_] = [self.metatrain_character_folders[i]]
            else:
                self.metatrain_dataset[cls_].append(self.metatrain_character_folders[i])

        # validation dataset
        self.metaval_dataset = {}
        for i, cls_ in enumerate(self.metaval_character_labels):
            if cls_ not in self.metaval_dataset.keys():
                self.metaval_dataset[cls_] = [self.metaval_character_folders[i]]
            else:
                self.metaval_dataset[cls_].append(self.metaval_character_folders[i])

        # testing dataset
        self.metatest_dataset = {}
        for i, cls_ in enumerate(self.metatest_character_labels):
            if cls_ not in self.metatest_dataset.keys():
                self.metatest_dataset[cls_] = [self.metatest_character_folders[i]]
            else:
                self.metatest_dataset[cls_].append(self.metatest_character_folders[i])
            
    # this function writes dataset to file
    def write2file(self, label_folder=os.path.join(dataset_path, 'labels')):
        # make directory
        os.system('mkdir -p {}'.format(label_folder))
        # label directories
        metatrain_label_fname = os.path.join(label_folder, 'metatrain_labels_{}way.txt'.format(self.num_classes))
        metaval_label_fname = os.path.join(label_folder, 'metaval_labels_{}way.txt'.format(self.num_classes))
        metatest_label_fname = os.path.join(label_folder, 'metatest_labels_{}way.txt'.format(self.num_meta_test_classes))
        
        # write to training label file
        with open(metatrain_label_fname, 'w') as f:
            for cls_ in range(self.num_classes):
                for folder_name_ in self.metatrain_dataset[cls_]:
                    f.write('{} {}\n'.format(cls_, folder_name_[len(dataset_path)+1:]))
                    
        # write to validation label file
        with open(metaval_label_fname, 'w') as f:
            for cls_ in range(self.num_classes):
                for folder_name_ in self.metaval_dataset[cls_]:
                    f.write('{} {}\n'.format(cls_, folder_name_[len(dataset_path)+1:]))
                    
        # write to testing label file
        with open(metatest_label_fname, 'w') as f:
            for cls_ in range(self.num_meta_test_classes):
                for folder_name_ in self.metatest_dataset[cls_]:
                    f.write('{} {}\n'.format(cls_, folder_name_[len(dataset_path)+1:]))

# main function
if __name__ == '__main__':
    n_way=5
    prep_data = PrepData(num_classes=n_way, num_meta_test_classes=n_way)
    prep_data.write2file()
    print('Done!')