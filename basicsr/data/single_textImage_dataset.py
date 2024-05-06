from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import unicodeDict_from_pickle
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SingleTextImageDataset(data.Dataset):
    """Read only lq images in the test phase, with text label
    Note: For Scene Text Super Resolution task. Label stores in meta_info_file. Needs a unicode mapping file to convert label to tensor.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There is one mode:
    1. 'meta_info_file': Use meta information file to generate paths.

    Args:
         opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            unicode_mapping_file (str): Path for unicode mapping file.
            meta_info_file (str): Path for meta information file, as well as image label.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleTextImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        self.ucode_dict


        # Generate paths
        if 'meta_info_file' in self.opt:
            self.paths = []
            self.labels = []
            with open(self.opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    path = osp.join(self.lq_folder, line.rstrip().split(' ')[0])
                    label = line.rstrip().split(' ')[1]
                    self.paths.append(path)
                    self.labels.append(label)
        else:
            raise ValueError('Need meta_info_file to generate paths.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'lq_path': lq_path, 'label': self.labels[index]}


