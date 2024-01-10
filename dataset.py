
import torch 
import numpy as np 

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        print(f"initializing {__class__.__name__}")
        
        self.data = np.arange(20).tolist() 
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        return self.data[idx]
    

if __name__ == '__main__':
    import argparse
    import time 

    import torch

    from dist import init_distributed_mode, get_rank

    args = argparse.ArgumentParser()
    args.add_argument('--dist-url')
    args.add_argument('--local-rank')

    args = args.parse_args()
    init_distributed_mode(args, False)

    print(args)

    device_ids = [1, 2]
    batch_size = 4
    num_workers = len(device_ids)*4
    epochs = 3

    train_dataset = SimpleDataset()
    val_dataset = SimpleDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=int(batch_size/args.world_size),
                                                shuffle=False,
                                                num_workers=int(num_workers/args.world_size),
                                                sampler=train_sampler)

    # train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, int(batch_size/args.world_size), drop_last=True)
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
    #                                                 batch_sampler=train_batch_sampler, 
    #                                                 num_workers=num_workers
    #                                             )

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=int(batch_size/args.world_size),
                                                shuffle=False,
                                                num_workers=int(num_workers/args.world_size),
                                                sampler=val_sampler,
                                                pin_memory=True
                                                )

    for epoch in range(1, epochs):
        train_sampler.set_epoch(epoch)
        
        print(f"EPOCH: {epoch}")
        print(f"=== train_dataloader for rank({get_rank()}):")
        train_batch_data = {}
        train_batch_data[get_rank()] = []
        tic = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            train_batch_data[get_rank()] += batch.numpy().tolist()
            print(f"rank({get_rank()} > batch: {batch_idx} > {batch}")
            
        print(">>> train-batch-data: ", train_batch_data)
        print(f"cost {time.time() - tic} ms")

        print(f"=== val_dataloader for rank({get_rank()}):")
        val_batch_data = {}
        val_batch_data[get_rank()] = []
        tic = time.time()
        for batch_idx, batch in enumerate(val_dataloader):
            val_batch_data[get_rank()] += batch.numpy().tolist()
            print(f"rank({get_rank()} > batch: {batch_idx} > {batch}")
            
        print(">>> val-batch-data: ", val_batch_data)
        print(f"cost {time.time() - tic} ms")
        
        print("====================================================================================")