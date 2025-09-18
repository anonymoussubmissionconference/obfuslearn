from torch.utils.data import DataLoader, WeightedRandomSampler
from MultiViewImageDataset import MultiViewImageDataset



class Multi_View_Loaders:
  def __init__(self, batch_size, transform, target_transform=None):
    self.batch_size = batch_size
    self.transform = transform
    self.train_loader, self.test_loader = self.train_test_loader(batch_size, transform)

  def train_test_loader(self, batch_size, transform):
      train_csvs = r"./division/malfiner/train0.11.csv"
      test_csvs = r"./division/malfiner/test0.11.csv"
      img_dirs = ["./features/malfinerssd", "../features/apis/malfiner-apisequence-128"]
      weights_files = r"./division/malfiner/weights0.11.csv"

      train_dataset = MultiViewImageDataset(annotation_file=train_csvs, image_dirs=img_dirs,
                                         weights_file=weights_files,
                                         transform=transform)
      test_dataset = MultiViewImageDataset(annotation_file=test_csvs, image_dirs=img_dirs,
                                        transform=transform)
      weights_tensor = train_dataset.get_weights_tensor()
      if weights_tensor is not None:
          print(weights_tensor[:10])
          sampler = WeightedRandomSampler(weights_tensor / weights_tensor.sum(), len(weights_tensor))
          train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
      else:
          train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=batch_size)

      return train_loader, test_loader

  def get_train_loader(self):
        return self.train_loader

  def get_test_loader(self):
       return self.test_loader