import os
import argparse
import yaml
import glob 

import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Resize
from torchaudio.transforms import AmplitudeToDB
from torchmetrics.classification import MultilabelF1Score
from tqdm import trange

from anuraset import AnuraSet
from models import ResNetClassifier
from time import time

from util import init_seed, min_max_normalize

from fvcore.nn import flop_count

def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = ResNetClassifier(model_type=cfg['model_type'])        # create an object instance of our CustomResNet18 class
    folder_name = cfg['folder_name']

    # load latest model state
    model_states = glob.glob(f'baseline/model_states/{folder_name}/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace(f'baseline/model_states/{folder_name}','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'baseline/model_states/{folder_name}{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch

def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    folder_name = cfg['folder_name']
    os.makedirs(f'baseline/model_states/{folder_name}', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'baseline/model_states/{folder_name}{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = f'baseline/model_states/{folder_name}config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

def train(model, data_loader, loss_fn, optimiser, metric_fn, device):
    
    num_batches = len(data_loader)
    model.train()
    ## running averages
    loss_total, metric_total = 0.0, 0.0
    size = len(data_loader.dataset) 
    epoch_flops_total = 0

    start_time = time() #start counting boye
    initial_allocated_memory = torch.cuda.memory_allocated()

    progressBar = trange(len(data_loader), leave=False)
    for batch_idx, (input, target, index) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)

        flop_input = (input, )
        flop_results = flop_count(model, flop_input)
        flops_dict = flop_results[0]  # Access the first element of the tuple containing the dictionary
        flops = sum(flops_dict.values())
        epoch_flops_total += flops

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # log statistics
        # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor
        loss_total += loss.item()  
        # log metrics
        metric = metric_fn(prediction, target)
        metric_total += metric.item()

        progressBar.set_description(
                '[Train] Loss: {:.4f}; F1-score macro: {:.4f} [{:>5d}/{:>5d}]'.format(
                    loss_total/(batch_idx+1),
                    metric_total/(batch_idx+1),
                    (batch_idx + 1) * len(input),
                    size
                )
            )
        progressBar.update(1)

    progressBar.close()  # end of epoch; finalize
    end_time = time() #click the stopwatch
    epoch_time = end_time - start_time
    print("Memory allocated after epoch:", torch.cuda.memory_allocated() - initial_allocated_memory)
    print(f"Training time for epoch: {epoch_time:.2f} seconds")
    



    # you should avoid last batch due different size

    loss_total /= num_batches   # shorthand notation for: loss_total = loss_total / len(dataLoader)
    metric_total /= num_batches
    
    print(f"FLOPs per epoch Training: {epoch_flops_total}")

    return loss_total, metric_total, epoch_flops_total

def validate(model, data_loader, loss_fn, metric_fn, device):

    '''
        Validation function. Looks like training
        function, except that we don't use any optimizer or gradient steps.
    '''
    start_time = time() 

    num_batches = len(data_loader)
    
    # put the model into evaluation mode
    model.eval()

    # running averages # correct
    loss_total, metric_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(data_loader), leave=False)

    initial_allocated_memory = torch.cuda.memory_allocated()
    epoch_flops_total = 0
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster

        for batch_idx, (input, target, index) in enumerate(data_loader):
            # put data and labels on device
            input, target = input.to(device), target.to(device)

            flop_input = (input, )
            flop_results = flop_count(model, flop_input)
            flops_dict = flop_results[0]  # Access the first element of the tuple containing the dictionary
            flops = sum(flops_dict.values())
            epoch_flops_total += flops

            prediction = model(input)
            loss = loss_fn(prediction, target)

            # log statistics
            loss_total += loss.item()
            # log metrics
            metric = metric_fn(prediction, target)
            metric_total += metric.item()

            progressBar.set_description(
                '[Validation] Loss: {:.4f}; F1-score macro: {:.4f}'.format(
                    loss_total/(batch_idx+1),
                    metric_total/(batch_idx+1),
                )
            )
            progressBar.update(1)

        progressBar.close()  # end of epoch; finalize
        end_time = time() #click the stopwatch
        epoch_time = end_time - start_time
        print("Memory allocated after epoch:", torch.cuda.memory_allocated() - initial_allocated_memory)
        print(f"Validation time for epoch: {epoch_time:.2f} seconds")
        #print(f"Total FLOPs during validation epoch: {epoch_flops_total}")

    loss_total /= num_batches
    metric_total /= num_batches
    print(f"FLOPs per epoch Validation: {epoch_flops_total}")

    return loss_total, metric_total, epoch_flops_total

def main():



    parser = argparse.ArgumentParser(description='Domain shift.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet152.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
  
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    device = torch.device(cfg['device'])
    print(f"Using device {device}")

    # Define Transformation

    resamp = torchaudio.transforms.Resample(
            orig_freq=22050,
            new_freq=16000
            )
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=512,
        hop_length=128,
        n_mels=128
        )
    time_mask = torchaudio.transforms.TimeMasking(
        time_mask_param=60, # mask up to 60 consecutive time windows
    )
    freq_mask = torchaudio.transforms.FrequencyMasking(
        freq_mask_param=8, # mask up to 8 consecutive frequency bins
    )
    train_transform = nn.Sequential(            # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            #resamp,                             # resample to 16 kHz
            mel_spectrogram,                    # convert to a spectrogram
            AmplitudeToDB(),                    # Turn a spectrogram from the power/amplitude scale to the decibel scale.
            # Normalize(),                      # normalize so min is 0 and max is 1
            time_mask,                          # randomly mask out a chunk of time
            freq_mask,                          # randomly mask out a chunk of frequencies
            Resize(cfg['image_size']),
                                    )
    val_transform = nn.Sequential(                           # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
            #resamp,                                             # resample to 16 kHz
            mel_spectrogram,                                            # convert to a spectrogram
            torchaudio.transforms.AmplitudeToDB(),
            #torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
            Resize(cfg['image_size']),
                                )

    ANNOTATIONS_FILE = os.path.join(
        cfg['data_root'],
        'metadata.csv'
    )
    
    AUDIO_DIR = os.path.join(
        cfg['data_root'],
        'audio'
    )
        
    training_data = AnuraSet(
            annotations_file=ANNOTATIONS_FILE, 
            audio_dir=AUDIO_DIR, 
            transformation=train_transform,
            train=True
                        )
    print(f"There are {len(training_data)} samples in the training set.")

    # TODO: call val not test!
    val_data = AnuraSet(
            annotations_file=ANNOTATIONS_FILE, 
            audio_dir=AUDIO_DIR, 
            transformation=val_transform,
            train=False
                            )
    print(f"There are {len(val_data)} samples in the test set.")

    train_dataloader = DataLoader(training_data, 
                                batch_size=cfg['batch_size'],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=4,
                                )
                                
    val_dataloader = DataLoader(val_data, 
                                batch_size=cfg['batch_size'],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=4,
                                )

    multi_label=cfg['multilabel']   
    if multi_label:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    


    # initialize model
    model_instance, current_epoch = load_model(cfg)
    
    model_instance.to(device)
    
    optimiser = torch.optim.Adam(model_instance.parameters(),
                                lr=cfg['learning_rate'])
    
    metric_fn = MultilabelF1Score(num_labels=cfg['num_classes']).to(device)
    
    start = time()
    progress_bar_epoch = trange(cfg['num_epochs'])
    print('Starting training')
    for current_epoch in range(current_epoch, cfg['num_epochs']):
        start_epoch = time()
        current_epoch += 1

        loss_train, metric_train, test_train = train(model_instance, train_dataloader, 
                                         loss_fn, optimiser, metric_fn, device)
        loss_val, metric_val, test_val = validate(model_instance, val_dataloader, 
                                        loss_fn, metric_fn, device)
        
        end_epoch = time()  # Measure the end time of the current epoch
        epoch_time = end_epoch - start_epoch  # Calculate the elapsed time for the epoch

        progress_bar_epoch.update(1)
        progress_bar_epoch.write(
        'Epoch: {:.0f}: Loss val: {:.4f} ; F1-score macro val: {:.4f} - Epoch time: {:.1f}s; Total time: {:.1f}s - {:.0f}%'.format(
            (current_epoch),
            loss_val,
            metric_val,
            epoch_time,
            (time()-start_epoch),
            (time()-start),
            (100*(current_epoch)/cfg['num_epochs'])
        )
        )

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'metric_train': metric_train,
            'metric_val': metric_val
        }
        # TODO: wandb or YAML?
        save_model(cfg, current_epoch, model_instance, stats)
    progress_bar_epoch.close()
    print("Finished training")

    # save model
    torch.save(model_instance.state_dict(), 
               'baseline/model_states/'+cfg['folder_name']+'final.pth')
               
    print('sTrained feed forward net saved at ' +
        'baseline/model_states/'+cfg['folder_name']+'final.pth')

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    main()