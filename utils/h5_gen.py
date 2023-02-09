import h5py as h5
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

data_root = ""
img_size = 64


class H5Data(Dataset):
    def __init__(self, root):
        super(H5Data, self).__init__()
        self.transforms = transforms.Compose([transforms.PILToTensor()])
        self.data = ImageFolder(root=root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.transforms(img), int(label)


def make_hdf5():
    dataset = H5Data(data_root)
    dataloader = DataLoader(dataset, batch_size=512, num_workers=4)
    images = None
    labels = None
    for x, y in tqdm(dataloader):
        images = np.concatenate(
            (images, np.transpose(x.numpy(), (0, 2, 3, 1)))) if images is not None else np.transpose(x.numpy(),
                                                                                                     (0, 2, 3, 1))
        labels = np.concatenate((labels, y.numpy())) if labels is not None else y.numpy()

    file_path = data_root + ".hdf5"

    with h5.File(file_path, "w") as f:
        f.create_dataset("imgs", data=images)
        f.create_dataset("labels", data=labels)
    return file_path


def print_hdf5(path, n=1):
    with h5.File(path, "r") as f:
        for i in range(n):
            print(f["imgs"][i])
            print(f["labels"][i])


if __name__ == '__main__':
    make_hdf5()
