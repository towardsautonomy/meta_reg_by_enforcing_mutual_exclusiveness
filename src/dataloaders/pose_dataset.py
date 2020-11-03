import numpy as np
import os
import random
from scipy import misc
import imageio
import pickle
import matplotlib.pyplot as plt

class DataGenerator(object):
  """
  Data Generator capable of generating batches of Omniglot data.
  A "class" is considered a class of omniglot digits.
  """

  def __init__(self, num_objects, num_samples_per_object, num_meta_objects, num_meta_test_samples_per_object, config={}):
    """
    Args:
      num_classes: Number of classes for classification (K-way)
      num_samples_per_class: num samples to generate per class in one batch
      num_meta_test_classes: Number of classes for classification (K-way) at meta-test time
      num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
      batch_size: size of meta batch size (e.g. number of functions)
    """
    self.num_objects = num_objects
    self.num_samples_per_object = num_samples_per_object
    self.num_meta_objects = num_meta_objects
    self.num_meta_test_samples_per_object = num_meta_test_samples_per_object

    data_folder = config.get('data_folder', 'datasets/pascal3d_pose')
    self.img_size = config.get('img_size', (128, 128))

    self.dim_input = np.prod(self.img_size)
    self.dim_output = 1

    # pickle file names
    train_pickle = os.path.join(data_folder, 'train_data.pkl')
    val_pickle = os.path.join(data_folder, 'val_data.pkl')
    test_pickle = os.path.join(data_folder, 'test_data.pkl')

    # load data
    train_data = pickle.load(open(train_pickle, 'rb'))
    val_data = pickle.load(open(val_pickle, 'rb'))
    test_data = pickle.load(open(test_pickle, 'rb'))

    # training data
    [train_X, train_y] = train_data
    self.train_X = self.normalize_img(np.array(train_X))
    self.train_y = np.array(train_y)

    # validation data
    [val_X, val_y] = val_data
    self.val_X = self.normalize_img(np.array(val_X))
    self.val_y = np.array(val_y)

    # test data
    [test_X, test_y] = test_data
    self.test_X = self.normalize_img(np.array(test_X))
    self.test_y = np.array(test_y)

  def normalize_img(self, image):
    image = image.astype(np.float32) / 255.0
    return image
    
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
      X_ = self.train_X
      y_ = self.train_y
      num_objects = self.num_objects
      num_samples_per_object = self.num_samples_per_object
    elif batch_type == "meta_val":
      X_ = self.val_X
      y_ = self.val_y
      num_objects = self.num_objects
      num_samples_per_object = self.num_samples_per_object
    else:
      X_ = self.test_X
      y_ = self.test_y
      num_objects = self.num_meta_objects
      num_samples_per_object = self.num_meta_test_samples_per_object
    y_ = y_[:,:,-1] # only the yaw angle. other two are constants
    y_ = y_[..., np.newaxis]
    all_image_batches, all_label_batches = [], []
    for i in range(batch_size):
      task_indices = random.sample(range(X_.shape[0]), num_objects)

      # meta-train dataset
      images = np.zeros((num_objects, num_samples_per_object, self.img_size[0], self.img_size[1] ))
      labels = np.zeros((num_objects, num_samples_per_object, self.dim_output ))
      for i, task_idx_ in enumerate(task_indices):
          samples_indices = random.sample(range(X_[task_idx_].shape[0]), num_samples_per_object)
          # meta-training dataset
          images[i] = X_[task_idx_][samples_indices]
          labels[i] = y_[task_idx_][samples_indices]
        
      images = np.reshape(images, (num_objects, num_samples_per_object, -1 ))

      all_image_batches.append(images)
      all_label_batches.append(labels)
    all_image_batches = np.stack(all_image_batches)
    all_label_batches = np.stack(all_label_batches)
    return all_image_batches, all_label_batches

# main function
if __name__ == '__main__':
  """Test the Data Generator"""
  dgen = DataGenerator(num_objects=4, num_samples_per_object=3, num_meta_objects=4, num_meta_test_samples_per_object=3)

  # get a meta training batch
  imgs, labels = dgen.sample_batch(batch_type='meta_train', batch_size=4, shuffle=True)
  imgs = np.reshape(imgs, newshape=(-1, dgen.num_objects, dgen.num_samples_per_object, dgen.img_size[0], dgen.img_size[1]))
  print('Images Shape : {} | Labels Shape : {}'.format(imgs.shape, labels.shape))

  # show images
  plt.figure(figsize=(10,20))
  plt.suptitle('Meta Training Data')
  for i in range(dgen.num_objects):
      for j in range(dgen.num_samples_per_object):
        plt.subplot(dgen.num_objects, dgen.num_samples_per_object, i*dgen.num_samples_per_object+j+1)
        plt.imshow(imgs[0, i, j], cmap='gray')
        plt.title('pose={:.2f}'.format(labels[0][i][j][0]))

  # get a meta testing batch
  imgs, labels = dgen.sample_batch(batch_type='meta_test', batch_size=4, shuffle=True)
  imgs = np.reshape(imgs, newshape=(-1, dgen.num_meta_objects, dgen.num_meta_test_samples_per_object, dgen.img_size[0], dgen.img_size[1]))
  # print('Images Shape : {} | Labels Shape : {}'.format(imgs.shape, labels.shape))

  # show images
  plt.figure(figsize=(10,20))
  plt.suptitle('Meta Testing Data')
  for i in range(dgen.num_meta_objects):
      for j in range(dgen.num_meta_test_samples_per_object):
        plt.subplot(dgen.num_meta_objects, dgen.num_meta_test_samples_per_object, i*dgen.num_meta_test_samples_per_object+j+1)
        plt.imshow(imgs[0, i, j], cmap='gray')
        plt.title('pose={:.2f}'.format(labels[0][i][j][0]))
  plt.show()