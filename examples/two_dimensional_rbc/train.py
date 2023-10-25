import os
import random
import sys
import importlib
import yaml

import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.transforms import Lambda

sys.path.append(os.getcwd()+'/../../')   # adapt to path where ae_class.py is situated
from ae_class import AE

def run(config):
    
    # -------------------------------------
    # 1. Set up AE parameters
    # -------------------------------------
    print(config)
    
    ae = AE(config)

    cuda = ae.cuda
    device = ae.device
    
    sys.path.append(ae.dataset_path)
    sys.path.append(ae.model_path)

    nn_dataset = importlib.import_module(ae.dataset_name)
    nn_module = importlib.import_module(ae.model_name)


    torch.manual_seed(ae.randomSeed)
    
    if sys.platform != "darwin":
        torch.cuda.manual_seed_all(ae.randomSeed)
    else:
        torch.mps.manual_seed(ae.randomSeed)
        
    np.random.seed(ae.randomSeed)
    random.seed(ae.randomSeed)
    
    print("""batch_size {0}, 
             num_epochs {1}, 
             encoding_dim {2}, 
             init. learning_rate {3:.2e}, 
             learning_rate lambda {4:.3f}, 
             weight_decay {5:.2e},
             p_conv {6:.2f}""".format(ae.batch_size, 
                                      ae.num_epochs, 
                                      ae.encoding_dim, 
                                      ae.learning_rate, 
                                      ae.lr_lambda, 
                                      ae.weight_decay,
                                      ae.p_conv))
    

    # -------------------------------------
    # 2. Datasets & -loader
    # -------------------------------------
    x_min,x_max = ae.get_minmax_values()
    transform_minmax = Lambda(lambda x:(x - x_min)/(x_max - x_min))  

    dataset_train = nn_dataset.RBC2dDataset(filepath=ae.datapath, 
                                            tstart=0,
                                            tend=ae.trainingLength,
                                            transform=transform_minmax)
    dataset_val   = nn_dataset.RBC2dDataset(filepath=ae.datapath, 
                                            tstart=ae.trainingLength, 
                                            tend=ae.trainingLength+ae.validationLength, 
                                            transform=transform_minmax)

    dataloader_train = DataLoader(dataset=dataset_train, 
                                  batch_size=ae.batch_size, 
                                  shuffle=ae.shuffle_train, 
                                  num_workers =  int(4*torch.cuda.device_count()), 
                                  drop_last=True,
                                  pin_memory=True)

    dataloader_val = DataLoader(dataset=dataset_val, 
                                batch_size=ae.batch_size, 
                                shuffle=ae.shuffle_val,
                                num_workers=int(4*torch.cuda.device_count()),
                                drop_last=True, 
                                pin_memory=True)
    
    # -------------------------------------
    # 3. Compile Model
    # -------------------------------------
    model = nn_module.Autoencoder(config)
    
    print(summary(model, (ae.nfields, ae.ny, ae.nx), device="cpu"))

    model = model.to(cuda)

    if sys.platform != "darwin":
        # DataParallel copies model to each GPU (on single machine) and scatters input data on batch dim across GPUs
        if torch.cuda.device_count() > 1:
            device_ids = [ii for ii in range(torch.cuda.device_count())]
            model      = nn.DataParallel(model, device_ids=device_ids)

    mse  = nn.MSELoss()
    loss = nn.MSELoss() # adapt to desired training loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=ae.learning_rate, weight_decay=ae.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: ae.lr_lambda)

    # -------------------------------------
    # 4. Train Model
    # -------------------------------------
    print('Starting Training...')
    time_start = time.time()

    loss_train_list, mse_train_list = [], []
    loss_val_list, mse_val_list = [], []
    min_val_loss = torch.tensor(float("inf"))
    epochs_no_improve = 0

    optimizer.zero_grad()
    num_epochs_run = 0
    for epoch in range(ae.num_epochs):

        # ------------------
        # 4.1 Training
        # ------------------
        model.train()
        
        train_mse = []
        train_loss = []
        for i, data in enumerate(dataloader_train):
            if ae.use_cuda:
                data = data.to(cuda)

            optimizer.zero_grad()
            outputs, _ = model(data)
            
            mse_train  = mse(outputs, data)
            loss_train = loss(outputs, data)

            loss_train.backward(retain_graph=True)
            optimizer.step()
            train_mse.append(mse_train.item())
            train_loss.append(loss_train.item())

        mse_train_list.append(np.mean(train_mse))
        loss_train_list.append(np.mean(train_loss))


        # ------------------
        # 4.2 Validation
        # ------------------
        model.eval()
        with torch.no_grad():
            val_mse = []
            val_loss = []
            for i,data_val in enumerate(dataloader_val):
                if ae.use_cuda:
                    data_val = data_val.to(cuda)

                outputs_val, _ = model(data_val)

                mse_val  = mse(outputs_val, data_val)
                loss_val = loss(outputs_val, data_val)

                val_mse.append(mse_val.item())
                val_loss.append(loss_val.item())

        mse_val_list.append(np.mean(val_mse))
        loss_val_list.append(np.mean(val_loss))

        print('Epoch {0}/{1}, learning rate {2:.3e} Train. MSE {3:.2e}, Val. MSE {4:.2e}, Train. Loss {5:.2e}, Val. Loss {6:.2e}'.format(epoch+1, 
                                                                                                                                         ae.num_epochs, 
                                                                                                                                         scheduler.get_last_lr()[0], 
                                                                                                                                         mse_train_list[-1], 
                                                                                                                                         mse_val_list[-1], 
                                                                                                                                         loss_train_list[-1], 
                                                                                                                                         loss_val_list[-1]))
        
        ae.save_history({'train_loss': loss_train_list, 
                         'val_loss':   loss_val_list, 
                         'train_mse':  mse_train_list, 
                         'val_mse':    mse_val_list})
        scheduler.step()

        # ------------------
        # Early stopping
        # ------------------
        if loss_val_list[-1] < min_val_loss:
            ae.save_model_checkpoint(deepcopy(model))
            epochs_no_improve = 0
            min_val_loss = loss_val_list[-1]
        else:
            epochs_no_improve += 1

        if epoch > 5 and epochs_no_improve == ae.n_epoch_stop:
            print("Early stopping after {0} epochs of no improvement".format(ae.n_epoch_stop))
            break

        num_epochs_run +=1
        
    time_end = time.time()
    print('Done. Total duration {0:.2f} h.\n{1:.2f} min per epoch'.format((time_end - time_start) / 3600,
                                                                          (time_end - time_start) / (60 * num_epochs_run)))

    
    
    

    
if __name__ == '__main__':
    
    cfg_fname = 'train_config.yaml' if len(sys.argv) <= 1 else sys.argv[1]
    with open(cfg_fname, "r") as f:
        config = yaml.safe_load(f)
    run(config)
    
# --------------------------------------------------------------------------------
#                                   END OF PROGRAM
# --------------------------------------------------------------------------------
