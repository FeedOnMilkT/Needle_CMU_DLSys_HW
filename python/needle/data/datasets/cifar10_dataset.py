import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms = transforms)
        self.transforms = transforms
        self.p = p

        X_list = []
        y_list = []

        if train:
          batch_files = [                
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
          ]
        else:
          batch_files = ["test_batch"]

        for file_name in batch_files:
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, "rb") as f:
                data_dict = pickle.load(f, encoding="bytes")

            X = data_dict[b"data"]                       # shape: (N, 3072)
            y = np.array(data_dict[b"labels"])          # shape: (N,)

            X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

            X_list.append(X)
            y_list.append(y)

        self.X = np.concatenate(X_list, axis = 0)
        self.y = np.concatenate(y_list, axis = 0)          
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img = self.X[index]
        label = self.y[index]

        if self.transforms is not None:
          img = self.apply_transfors(img)

        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
