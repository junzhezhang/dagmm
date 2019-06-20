import os
import argparse
from solver import Solver
# from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from torch.utils.data import DataLoader

class NSLKDDLoader(object):
    def __init__(self, train_path, test_path, mode="train"):
        #TODO(junzhe) to think abt the mode settings.
        self.mode=mode

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        train_labels = train_data["nsl_kdd_train"][:,-1]
        train_features = train_data["nsl_kdd_train"][:,:-1]

        test_labels = test_data["nsl_kdd_test"][:,-1]
        test_features = test_data["nsl_kdd_test"][:,:-1]

        train_attack_features = train_features[train_labels==0]
        train_attack_labels = train_labels[train_labels==0]

        self.train = train_attack_features
        self.train_labels = train_attack_labels

        self.test = test_features
        self.test_labels = test_labels

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])


def get_nsl_kdd_loader(train_path, test_path, batch_size, mode='train'):
    #TODO(junzhe) to think abt the mode settings.
    """Build and return data loader."""

    dataset = NSLKDDLoader(train_path, test_path, mode)
    #TODO(junzhe) disbale shuffle, same as random seed.
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)

    data_loader = get_nsl_kdd_loader(config.train_path, config.test_path, batch_size=config.batch_size, mode=config.mode)
    
    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4)


    # Training settings
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--train_path', type=str, default='nsl_kdd_train.npz')
    parser.add_argument('--test_path', type=str, default='nsl_kdd_test.npz')
    parser.add_argument('--log_path', type=str, default='./dagmm/logs')
    parser.add_argument('--model_save_path', type=str, default='./dagmm/models')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=124)
    parser.add_argument('--model_save_step', type=int, default=124)
    parser.add_argument('--input_dimension',type=int,default=122)
    parser.add_argument('--anomaly_percentage',type=int,default=20)
    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
