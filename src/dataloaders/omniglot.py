import random
import glob
import numpy as np
import os
from scipy import misc
import imageio
import matplotlib.pyplot as plt
    
def image_file_to_array(filename, dim_input):
  """
  Takes an image path and returns numpy array
  Args:
    filename: Image filename
    dim_input: Flattened shape of image
  Returns:
    1 channel image
  """
  image = imageio.imread(filename)
  image = image.reshape([dim_input])
  image = image.astype(np.float32) / 255.0
  image = 1.0 - image
  return image


class DataGenerator(object):
    def __init__(self, num_classes, num_samples_per_class, num_meta_test_classes, num_meta_test_samples_per_class, config={}):
        """
        Args:
          num_samples_per_class: num samples to generate per class in one batch
          num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
        """
        
        # get config
        self.labels_folder = config.get('labels_folder', 'labels')
        self.data_folder = config.get('data_folder', 'datasets/omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        # number of classes
        self.num_classes = num_classes
        self.num_meta_test_classes = num_meta_test_classes
        # input and output dimensions
        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes
        # number of samples
        self.num_samples_per_class = num_samples_per_class
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        
        # label directories
        metatrain_label_fname = os.path.join(self.labels_folder, 'metatrain_labels_{}way.txt'.format(self.num_classes))
        metaval_label_fname = os.path.join(self.labels_folder, 'metaval_labels_{}way.txt'.format(self.num_classes))
        metatest_label_fname = os.path.join(self.labels_folder, 'metatest_labels_{}way.txt'.format(self.num_meta_test_classes))

        # make sure label files exist
        if not os.path.exists(metatrain_label_fname) or \
            not os.path.exists(metaval_label_fname) or \
            not os.path.exists(metatest_label_fname):
            raise('Label files do not exist')
            
        # read labels
        # training dataset
        self.metatrain_dataset = {}
        with open(metatrain_label_fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split()
                cls_ = int(label[0])
                character_folder_ = label[1]
                
                # add to the dataset
                if cls_ not in self.metatrain_dataset.keys():
                    self.metatrain_dataset[cls_] = [character_folder_]
                else:
                    self.metatrain_dataset[cls_].append(character_folder_)
                    
        # validation dataset
        self.metaval_dataset = {}
        with open(metaval_label_fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split()
                cls_ = int(label[0])
                character_folder_ = label[1]
                
                # add to the dataset
                if cls_ not in self.metaval_dataset.keys():
                    self.metaval_dataset[cls_] = [character_folder_]
                else:
                    self.metaval_dataset[cls_].append(character_folder_)
                    
        # testing dataset
        self.metatest_dataset = {}
        with open(metatest_label_fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split()
                cls_ = int(label[0])
                character_folder_ = label[1]
                
                # add to the dataset
                if cls_ not in self.metatest_dataset.keys():
                    self.metatest_dataset[cls_] = [character_folder_]
                else:
                    self.metatest_dataset[cls_].append(character_folder_)
                    
    def print_dataset_info(self):
        """
        Prints dataset information
        """
        n_train_folders = sum([len(self.metatrain_dataset[n]) for n in self.metatrain_dataset.keys()])
        n_val_folders = sum([len(self.metaval_dataset[n]) for n in self.metaval_dataset.keys()])
        n_test_folders = sum([len(self.metatest_dataset[n]) for n in self.metatest_dataset.keys()])
        print('Training Dataset   | Num of characters: {:4d}, Num of classes: {:2d}'.format(n_train_folders, len(self.metatrain_dataset.keys())))
        print('Validation Dataset | Num of characters: {:4d}, Num of classes: {:2d}'.format(n_val_folders, len(self.metaval_dataset.keys())))
        print('Testing Dataset    | Num of characters: {:4d}, Num of classes: {:2d}'.format(n_test_folders, len(self.metatest_dataset.keys())))

    def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
          batch_type: meta_train/meta_val/meta_test
          shuffle: randomly shuffle classes or not
          swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
          A a tuple of (1) Image batch and (2) Label batch where
          image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap is False
          where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "meta_train":
            dataset = self.metatrain_dataset
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "meta_val":
            dataset = self.metaval_dataset
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            dataset = self.metatest_dataset
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class

        images_batch = []
        labels_batch = []
        for i in range(batch_size):
            images_this_batch = []
            labels_this_batch = []
            for j in range(num_classes):
                # sample one character folder
                character_folder_sampled = random.sample(dataset[j], 1)[0]
                character_imfnames = glob.glob(os.path.join(self.data_folder, character_folder_sampled, '*.png'))
                # sample 'num_samples_per_class' characters
                character_imfnames = random.sample(character_imfnames, num_samples_per_class)
                
                images = [image_file_to_array(character_imfname_, self.dim_input) for character_imfname_ in character_imfnames]
                images_this_batch.append(images)
                # labels
                labels = np.zeros(num_classes)
                labels[j] = 1.0
                labels_this_batch.append([labels] * num_samples_per_class)
                
            # convert to numpy array
            images_this_batch = np.array(images_this_batch)
            labels_this_batch = np.array(labels_this_batch)
            # shuffle
            if shuffle == True:
                shuffler = np.random.permutation(num_classes)
                images_this_batch[:] = images_this_batch[shuffler]
                labels_this_batch[:] = labels_this_batch[shuffler]
                
            # images and labels batch
            images_batch.append(images_this_batch)
            labels_batch.append(labels_this_batch)
    
#         images_batch = np.reshape(images_batch, (batch_size, num_classes, num_samples_per_class, self.img_size[0], self.img_size[1]))
        images_batch = np.reshape(images_batch, (batch_size, num_classes, num_samples_per_class, -1))
        labels_batch = np.reshape(labels_batch, (batch_size, num_classes, num_samples_per_class, -1))
        
        # swap classes and num_samples axis
        if swap == True:
            images_batch = np.swapaxes(images_batch, 1, 2)
            labels_batch = np.swapaxes(labels_batch, 1, 2)
            
        return images_batch, labels_batch
    
# main function
if __name__ == '__main__':
    """Test the Data Generator"""
    dgen = DataGenerator(num_classes=5, num_samples_per_class=3, num_meta_test_classes=5, num_meta_test_samples_per_class=3)

    # print info
    dgen.print_dataset_info()

    # get a meta training batch
    imgs, labels = dgen.sample_batch(batch_type='meta_train', batch_size=4, shuffle=True)
    imgs = np.reshape(imgs, newshape=(-1, dgen.num_classes, dgen.num_samples_per_class, dgen.img_size[0], dgen.img_size[1]))
    # print('Images Shape : {} | Labels Shape : {}'.format(imgs.shape, labels.shape))

    # show images
    plt.figure(figsize=(8,20))
    plt.suptitle('Meta Training Data')
    for i in range(dgen.num_classes):
        for j in range(dgen.num_samples_per_class):
            plt.subplot(dgen.num_classes, dgen.num_samples_per_class, i*dgen.num_samples_per_class+j+1)
            plt.imshow(imgs[0, i, j], cmap='gray')
            plt.title('Class {}'.format(np.argmax(labels[0, i, j])))

    # get a meta testing batch
    imgs, labels = dgen.sample_batch(batch_type='meta_test', batch_size=4, shuffle=True)
    imgs = np.reshape(imgs, newshape=(-1, dgen.num_meta_test_classes, dgen.num_meta_test_samples_per_class, dgen.img_size[0], dgen.img_size[1]))
    # print('Images Shape : {} | Labels Shape : {}'.format(imgs.shape, labels.shape))

    # show images
    plt.figure(figsize=(8,20))
    plt.suptitle('Meta Testing Data')
    for i in range(dgen.num_meta_test_classes):
        for j in range(dgen.num_meta_test_samples_per_class):
            plt.subplot(dgen.num_meta_test_classes, dgen.num_meta_test_samples_per_class, i*dgen.num_meta_test_samples_per_class+j+1)
            plt.imshow(imgs[0, i, j], cmap='gray')
            plt.title('Class {}'.format(np.argmax(labels[0, i, j])))
    plt.show()