import os
import json
import time, random
import torch
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc, densenet121, AutoEncoder
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandAffined,
    RandFlipd,
    Resized,
)
import datetime
from src.utils import data_split, ensure_folder_exists
print_config()
# Set Determinism
set_determinism(seed=42)
pretrain_data_path = './data/MedNIST/pretrain_data_dicts.json'
assert os.path.exists(pretrain_data_path), 'Pretrain data not found'
logdir_path = os.path.join('./logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
ensure_folder_exists(logdir_path)
assert os.path.exists(logdir_path), 'Log directory not found'

with open(pretrain_data_path, "r") as json_f:
    json_data = json.load(json_f)

splited_data = data_split(json_data, [0.8, 0.2], seed=42)

train_data = splited_data["train"]
val_data = splited_data["val"]

print("Total Number of Training Data Samples: {}".format(len(train_data)))
print(train_data[1])
print("#" * 10)
print("Total Number of Validation Data Samples: {}".format(len(val_data)))
print(val_data[-1])
print("#" * 10)

# Define Training Transforms
holes = 10
hole_spatial_size = 10
train_transforms = Compose(
    [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        # Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=-57,
        #     a_max=164,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        # CropForegroundd(keys=["image"], source_key="image"),
        # SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),
        # RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=2),
        CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
        RandAffined(
            keys=["image", "image_2", "gt_image"],
            prob=0.8,
            rotate_range=0.5,
            shear_range=0.0,
            translate_range=0.0,
            scale_range=0.0,
            mode=("bilinear", "bilinear", "bilinear"),
            padding_mode="zeros",
        ),
        RandFlipd(keys=["image", "image_2", "gt_image"], prob=0.8, spatial_axis=0),
        RandFlipd(keys=["image", "image_2", "gt_image"], prob=0.8, spatial_axis=1),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=holes, spatial_size=hole_spatial_size, dropout_holes=True, fill_value=0
                ),
                RandCoarseDropoutd(
                    keys=["image"], prob=1.0, holes=holes, spatial_size=hole_spatial_size, dropout_holes=False, fill_value=0
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        # OneOf(
        #     transforms=[
        #         RandCoarseDropoutd(
        #             keys=["image_2"], prob=1.0, holes=holes, spatial_size=hole_spatial_size, dropout_holes=True, fill_value=0
        #         ),
        #         RandCoarseDropoutd(
        #             keys=["image_2"], prob=1.0, holes=holes, spatial_size=hole_spatial_size, dropout_holes=False, fill_value=0
        #         ),
        #     ]
        # ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=12
                           ),
        
        # Resized(keys=["image", "image_2"], spatial_size=(224, 224, 3), mode=("bilinear", "nearest")),
    ]
)

# check_ds = Dataset(data=train_data, transform=train_transforms)
# check_loader = DataLoader(check_ds, batch_size=1)
# check_data = first(check_loader)
# image = check_data["image"][0][0]
# image2 = check_data["image_2"][0][0]
# gt_image = check_data["gt_image"][0][0]
# print(f"image shape: {image.shape}")
# print(f"image2 shape: {image2.shape}")
# print(f"gt_image shape: {gt_image.shape}")

# plt.figure("check", (18, 6))
# plt.subplot(1, 3, 1)
# plt.title("image")
# plt.imshow(image, cmap="gray")
# plt.subplot(1, 3, 2)
# plt.title("image2")
# plt.imshow(image2, cmap="gray")
# plt.subplot(1, 3, 3)
# plt.title("gt_image")
# plt.imshow(gt_image, cmap="gray")
# plt.show()

# Define Network ViT backbone & Loss & Optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
model = AutoEncoder(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(4,8,16,32),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
# model = ViTAutoEnc(in_channels=1,
#                    img_size=64,
#                    patch_size=4,
#                    out_channels=1,
#                    spatial_dims=2).to(device)

model.eval()
with torch.no_grad():
    output = model(torch.rand(1, 1, 64, 64).to(device))
    print(output.shape)
    
# Training Config
# Define Hyper-paramters for training loop
max_epochs = 800
val_interval = 5
batch_size = 128
lr = 1e-4

epoch_loss_values = []
step_loss_values = []
epoch_cl_loss_values = []
epoch_recon_loss_values = []
val_loss_values = []
best_val_loss = 1000.0

recon_loss = L1Loss()
contrastive_loss = ContrastiveLoss(temperature=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Define DataLoader using MONAI, CacheDataset needs to be used
# train_ds = Dataset(data=train_data, transform=train_transforms)
train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

# val_ds = Dataset(data=val_data, transform=train_transforms)
val_ds = CacheDataset(data=val_data, transform=train_transforms, cache_rate=1)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    epoch_cl_loss = 0
    epoch_recon_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        start_time = time.time()

        inputs, inputs_2, gt_input = (
            batch_data["image"].to(device),
            batch_data["image_2"].to(device),
            batch_data["gt_image"].to(device),
        )
        optimizer.zero_grad()
        outputs_v1 = model(inputs)
        outputs_v2 = model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=-1)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=-1)

        r_loss = recon_loss(outputs_v1, gt_input)
        cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = r_loss + cl_loss * r_loss

        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        step_loss_values.append(total_loss.item())

        # CL & Recon Loss Storage of Value
        epoch_cl_loss += cl_loss.item()
        epoch_recon_loss += r_loss.item()

        end_time = time.time()
        # print(
        #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
        #     f"train_loss: {total_loss.item():.4f}, "
        #     f"time taken: {end_time-start_time}s"
        # )

    epoch_loss /= step
    epoch_cl_loss /= step
    epoch_recon_loss /= step

    epoch_loss_values.append(epoch_loss)
    epoch_cl_loss_values.append(epoch_cl_loss)
    epoch_recon_loss_values.append(epoch_recon_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        print("Entering Validation for epoch: {}".format(epoch + 1))
        total_val_loss = 0
        val_step = 0
        model.eval()
        for val_batch in val_loader:
            val_step += 1
            start_time = time.time()
            inputs, gt_input = (
                val_batch["image"].to(device),
                val_batch["gt_image"].to(device),
            )
            # print("Input shape: {}".format(inputs.shape))
            outputs = model(inputs)
            val_loss = recon_loss(outputs, gt_input)
            total_val_loss += val_loss.item()
            end_time = time.time()

        total_val_loss /= val_step
        val_loss_values.append(total_val_loss)
        print(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

        if total_val_loss < best_val_loss:
            print(f"Saving new model based on validation loss {total_val_loss:.4f}")
            best_val_loss = total_val_loss
            # checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(model.state_dict(), os.path.join(logdir_path, f"best_model_epoch_{epoch+1}.pth"))

        
        plt.figure(1, figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epoch_loss_values)
        plt.grid()
        plt.title("Training Loss")

        plt.subplot(2, 2, 2)
        plt.plot(val_loss_values)
        plt.grid()
        plt.title("Validation Loss")

        plt.subplot(2, 2, 3)
        plt.plot(epoch_cl_loss_values)
        plt.grid()
        plt.title("Training Contrastive Loss")

        plt.subplot(2, 2, 4)
        plt.plot(epoch_recon_loss_values)
        plt.grid()
        plt.title("Training Recon Loss")

        plt.savefig(os.path.join(logdir_path, "loss_plots.png"))
        plt.close(1)

print("Done")
# model.load_state_dict(torch.load(os.path.join(logdir_path, "best_metric_model.pth")))