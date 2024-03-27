import os.path
from tqdm import tqdm
from torch.utils.data import Dataset
import pathlib
from torchvision import transforms
import numpy as np
import  torch
from PIL import Image,ImageFile
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize([0.456, 0.485, 0.406], [0.224, 0.229, 0.225])])


import pathlib
import glob
from torch.utils.data import Dataset
class ImageFolderCustom(Dataset):

    
    def __init__(self, targ_dir,exclude_files, transform=None):

        

        self.paths = [file for file in pathlib.Path(targ_dir).rglob('*/*') if
                      not any(exclude_file in str(file) for exclude_file in exclude_files)]
        
        self.transform = transform
       
        self.class_to_idx = {'images10':0,'images01':1,'images11':2}

   
    def load_image(self, index: int) :
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    
    def __len__(self):
        "Returns the total number of samples."
        return len(self.paths)

    
    def __getitem__(self, index) :
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)





