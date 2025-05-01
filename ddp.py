"""Skeleton for using DDP"""

import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
import torch


def train():
    if global_rank == 0:
        """initialize weights and biases"""

    dataloader = Dataloader()
    model = MyModel()

    if os.path.exists("latest_checkpoint.pth"):  # load latest checkpoint
        # load optimizer step and other variables
        model.load_state_dict(torch.load("latest_checkpoint.pth"))

    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters, lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for data, labels in dataloader:
            loss = loss_fn(model(data), labels)  # forward step
            loss.backward()  # backward step + gradient sync
            optimizer.step()  # update weights
            optimizer.zero_grad()  # set gradients to zero

        if global_rank == 0:
            torch.save(model.state_dict(), "latest_checkpoint.pth")
            """collect checkpoint values"""


if __name__ == "__main__":
    # rank across GPUs in same network
    local_rank = int(os.environ["LOCAL_RANK"])
    # rank across all GPUs being used
    global_rank = int(os.environ["RANK"])

    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)  # set device to local_rank

    train()

    destroy_process_group()
