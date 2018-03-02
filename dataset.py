from PIL import Image
import os
import os.path
import random
import math

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, size=100000, transform=None,
                 loader=default_image_loader):
        self.base_path = datapath
        self.size = size
        self.data = {}
        self.pairs = []
        for index in open(os.path.join(self.base_path, "index.txt")):
            print ("reading index: ", index)
            index = index.strip()
            data = []
            for line in open(os.path.join(self.base_path, index, "index.txt")):
                data.append({'filename': line.rstrip('\n')})

            print ("number of images: ", len(data))
            i = 0
            for line in open(os.path.join(self.base_path, index, "fGPS.txt")):
                gps_info = line.rstrip('\n').split(",")
                data[i]['gps'] = [float(gps_info[0]), float(gps_info[1])]
                i = i + 1
                if (i >= len(data)):
                    break

            print ("number of gps info: ", i)
            self.data[index] = data
#            if (len(data) < self.size):
#                self.size = len(data)

        if os.path.exists(os.path.join(self.base_path, "pairs.txt")):
            for line in open(os.path.join(self.base_path, "pairs.txt")):
                pairs = line.rstrip('\n').split(",")
                self.pairs.append(((pairs[0], pairs[1]), (pairs[2], pairs[3])))
        else:
            self.make_pairs()
            pairs_file = open(os.path.join(self.base_path, "pairs.txt"), 'w')
            for pair in self.pairs:
                pairs_file.write("{},{},{},{}\n".format(pair[0][0], pair[0][1], pair[1][0], pair[1][1]))
            pairs_file.close()

        if (len(self.pairs) < size):
            self.size = len(self.pairs)

        self.transform = transform
        self.loader = loader

    def distance(self, gps1, gps2):
        return math.hypot(gps1[0] - gps2[0], gps1[1] - gps2[1])

    def find_arbitrary_match(self, anchor_gps, positive_data_index):
        shuffled_index = list(range(len(self.data[positive_data_index])))
        random.shuffle(shuffled_index)
        for i in shuffled_index:
            if (self.distance(self.data[positive_data_index][shuffled_index[i]]['gps'], anchor_gps) < 0.00002):
                return i
        return -1

    def make_pairs(self):
        import time
        for i in list(self.data.keys()):
            positive_keys = list(self.data.keys())
            positive_keys.remove(i)
            t1 = time.time()
            for anchor in self.data[i]:
                for positive_index in positive_keys:
                    closest_sample = self.data[positive_index][0]
                    min_distance = 100.
                    for sample in self.data[positive_index]:
                        distance = self.distance(anchor['gps'], sample['gps']) 
                        if (distance < min_distance):
                            min_distance = distance
                            closest_sample = sample
                    if min_distance < 0.0002:
                        self.pairs.append(((i, anchor['filename']), (positive_index, closest_sample['filename'])))
            t2 = time.time()
            print (t2-t1)
        return self.pairs

    def __getitem__(self, index):
        ((anchor_data_index, anchor_path), (positive_data_index, positive_path)) = self.pairs[index]
        negative_data_index = random.choice(list(self.data.keys()))
        negative_dict = random.choice(self.data[negative_data_index])
        negative_path = negative_dict['filename']

        anchor = self.loader(os.path.join(self.base_path, anchor_data_index, anchor_path))
        positive = self.loader(os.path.join(self.base_path, positive_data_index, positive_path))
        negative = self.loader(os.path.join(self.base_path, negative_data_index, negative_path))
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return self.size

    """
    def __getitem__(self, index):
        keys = list(self.data.keys())
        anchor_data_index = random.choice(keys)
        negative_data_index = random.choice(keys)
        keys.remove(anchor_data_index)
        positive_data_index = random.choice(keys)

        anchor_dict = self.data[anchor_data_index][index]
        negative_dict = random.choice(self.data[negative_data_index])
        positive_dict = None

        positive_index = self.find_arbitrary_match(anchor_dict['gps'], positive_data_index)
        if (positive_index == -1):
            print ("did not find a close image")
            positive_data_index = anchor_data_index
            if (index+3 < len(self.data[positive_data_index])):
                positive_dict = self.data[positive_data_index][index+3]
            else:
                positive_dict = self.data[positive_data_index][index-3]
        else:
            positive_dict = self.data[positive_data_index][positive_index]

        anchor = self.loader(os.path.join(self.base_path, anchor_data_index, anchor_dict['filename']))
        positive = self.loader(os.path.join(self.base_path, positive_data_index, positive_dict['filename']))
        negative = self.loader(os.path.join(self.base_path, negative_data_index, negative_dict['filename']))
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    """
