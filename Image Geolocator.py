import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import s2sphere as s2
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from torchvision import transforms
    from io import BytesIO
    import msgpack
    from PIL import Image
    from torch.utils.data import IterableDataset
    import geopandas as gpd
    import zipfile
    import os
    import csv
    import pandas as pd
    import torch.nn as nn
    import torch.nn.functional as F
    import hashlib
    import time

    return (
        BytesIO,
        DataLoader,
        F,
        Image,
        IterableDataset,
        csv,
        hashlib,
        mo,
        msgpack,
        nn,
        os,
        pd,
        plt,
        s2,
        time,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _(csv, msgpack, os):
    all_the_rows = []
    path = r"data"
    global_idx = 0
    if os.path.exists(os.path.join(path, "metadata.csv")):
        os.remove(os.path.join(path, "metadata.csv"))

    for file in os.listdir(path):
         with open(os.path.join(path, file), "rb") as f:
            data = msgpack.Unpacker(f, raw=False)
            file_name = f.name
            for idx, item in enumerate(iter(data)):
                d = {
                    "global_idx": global_idx,
                    "file_path": file_name,
                    "local_idx": idx,
                    "latitude": item['latitude'],
                    "longitude": item['longitude']
                }

                global_idx += 1

                all_the_rows.append(d)

    # for folder in os.listdir(path):
    #      with zipfile.ZipFile(os.path.join(path, folder)) as archive:
    #          with archive.open(archive.filelist[0].filename) as file:
    #             data = msgpack.Unpacker(file, raw=False)
    #             file_name = file.name

    #             for idx, item in enumerate(iter(data)):
    #                 d = {
    #                     "global_idx": global_idx,
    #                     "file_path": file_name,
    #                     "local_idx": idx,
    #                     "latitude": item['latitude'],
    #                     "longitude": item['longitude']
    #                 }

    #                 global_idx += 1

    #                 all_the_rows.append(d)


    with open(os.path.join(path, 'metadata.csv'), 'w', newline='') as output_file:
        keys = all_the_rows[0].keys()
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_the_rows)
    return (path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating an S2 cell grid
    """)
    return


@app.cell
def _(s2):
    # Creates an s2 cell grid using recursion, with more cells in areas with more photos
    def build_adaptive_grid(current_cell, cell_photos, max_level, max_photos):
        # Base cases
        # Stops if the current cell level reaches max level and returns
        if current_cell.level() >= max_level:
            return [current_cell]

        # Stops if the number of cells photos is less than the max number of photos and returns
        if len(cell_photos) <= max_photos:
            return [current_cell]

        # Recursion
        final_cells = []

        # Generates 4 child cells
        for i in range(4):
            child_cell = current_cell.child(i)

            # Creates a new "cell_photos" call child_photos. child_photos only contains cell_photos which fall into the child cell
            child_photos = [
                photo_id for photo_id in cell_photos 
                if child_cell.contains(photo_id)
            ]

            # If the child has 0 photos, ignore it
            if not child_photos:
                continue

            # Recurse into the child
            deep_cells = build_adaptive_grid(child_cell, child_photos, max_level, max_photos)
            final_cells.extend(deep_cells)

        return final_cells


    def generate_global_grid(raw_coordinates, max_level=15, max_photos=100):
        print(f"# of Photos: {len(raw_coordinates)}")

        # Converts coordinates to cell_ids
        photo_cell_ids = [
            s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, long)) for lat, long in raw_coordinates
        ]

        # Creates six base faces of Earth
        base_faces = [s2.CellId.from_face_pos_level(face, 0, 0) for face in range(6)]

        all_final_cells = []

        # Feed each face into the recursive function
        for face_cell in base_faces:

            # Filter photos for just this face
            face_photos = [p for p in photo_cell_ids if face_cell.contains(p)]

            if not face_photos:
                continue 

            # Runs the recursive builder for this face
            cells = build_adaptive_grid(face_cell, face_photos, max_level, max_photos)
            all_final_cells.extend(cells)

        print(f"Created grid of {len(all_final_cells)} cells/classes.")
        return all_final_cells

    return (generate_global_grid,)


@app.cell
def _(generate_global_grid, os, path, pd):
    coords = pd.read_csv(os.path.join(path, 'metadata.csv'))
    coords = list(zip(coords['latitude'], coords['longitude']))

    final_s2_cells = generate_global_grid(coords, max_level=2, max_photos=10000)

    cell_id_dict = dict(zip(final_s2_cells, list(range(len(final_s2_cells)))))

    cell_num = len(final_s2_cells)
    return cell_id_dict, cell_num


@app.cell
def _(cell_id_dict, s2):
    def get_class_id(row, s2_cells):
        # Convert lat/lng to Leaf Cell
        leaf = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(row['latitude'], row['longitude']))

        # Check which parent cell contains it
        for parent_cell in s2_cells:
            if parent_cell.contains(leaf):
                return cell_id_dict[parent_cell]
        return -1 # Fallback if something goes wrong


    return


@app.cell
def _():
    # This is not necessary anymore, and takes way too much time

    # Adds class id to CSV metadata
    # df = pd.read_csv(os.path.join(path, 'metadata.csv'))
    # df['s2_class_id'] = df.apply(lambda x: get_class_id(x, s2_cells = final_s2_cells), axis=1)

    # print(df) 

    # df.to_csv(os.path.join(path, 'metadata.csv'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the data
    """)
    return


@app.cell
def _(BytesIO, F, Image, IterableDataset, hashlib, msgpack, nn, os, s2, torch):
    # Class that makes a GeoDataset specifically for the geolocation use case. Inherits torch.utils.data.Dataset
    class GeoDataset(IterableDataset):
        def __init__(self, path, class_map, transform=None, split = 'train', split_ratio = 0.8, show_image = False):
            # path: file path to the file containing the metadata.csv and data folders. Initialized at the start.
            self.path = path
            # transform: A pytorch transform list which tells the 
            self.transform = transform

            self.class_map = class_map

            self.split = split
            self.split_ratio = split_ratio

            self.show_image = show_image

        def __iter__(self):
            for file_name in os.listdir(self.path):

                if not file_name.endswith('.msg'):
                    continue

                with open(os.path.join(self.path, file_name), "rb") as file:

                        unpacker = msgpack.Unpacker(file, raw=False)
                        for item in unpacker:

                            # 1. Deterministic Splitting Logic
                            # We hash the image bytes to get a consistent value between 0 and 1
                            data_hash = hashlib.md5(item['image']).hexdigest()
                            # Convert hex to a float between 0 and 1
                            val = int(data_hash, 16) / float(1 << 128)

                            if self.split == 'train':
                                if val > self.split_ratio: continue # Skip if in test range
                            else: # test/val
                                if val <= self.split_ratio: continue # Skip if in train range

                            lat, lng = item['latitude'], item['longitude']
                            leaf = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lng))

                            label_int = -1

                            for level in range(15, -1, -1):
                                parent_cell = leaf.parent(level)

                                if parent_cell in self.class_map:
                                    label_int = self.class_map[parent_cell]
                                    break

                            if label_int == -1:
                                continue # Skip if outside our grid


                            image = Image.open(BytesIO(item['image']))

                            # To show image, comment out these last two lines.
                            if self.show_image == False:
                                if self.transform != None:
                                    image = self.transform(image)

                            yield image, label_int

    class GeM(nn.Module):
        def __init__(self, p=3.0, eps=1e-6):
            super(GeM, self).__init__()

            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            x_p = x.clamp(min=self.eps).pow(self.p)

            pooled = F.avg_pool2d(x_p, (x_p.size(-2), x_p.size(-1)))

            return pooled.pow(1.0 / self.p)

    return (GeoDataset,)


@app.cell
def _(DataLoader, GeoDataset, cell_id_dict, path, transforms):
    # https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images/data
    # https://www.kaggle.com/code/pkompally/location-cnn

    transform = transforms.Compose([
        transforms.Resize((299, 299)), # Height, Width
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GeoDataset(path=path, class_map = cell_id_dict, transform=transform, split = 'train')

    test_dataset = GeoDataset(path=path, class_map = cell_id_dict, transform=transform, split = 'test')

    train_dataloader = DataLoader(train_dataset, batch_size = 64)
    test_dataloader = DataLoader(test_dataset, batch_size = 64)
    return test_dataloader, train_dataloader, transform


@app.cell
def _(cell_num, nn, torch, torchvision):
    # [Input Image] -> [Conv + ReLU + BN] -> [Pooling] -> ... -> [Global Average Pooling] -> [Dense Layer (Embedding)] -> [Softmax]

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights = weights)
    # model = torchvision.models.resnet18()

    # for param in model.parameters():
    #     param.requires_grad = False

    # model.avgpool = nn.Sequential(
    #     GeM(),
    #     nn.Flatten()
    # )
    model.fc = nn.Linear(2048, cell_num)
    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 1024),
    #     nn.BatchNorm1d(1024),
    #     nn.ReLU(),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(1024, cell_num)
    # )

    head_params = list(model.fc.parameters())

    head_param_ids = [id(p) for p in head_params]
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.AdamW([
    #     #{'params': backbone_params, 'lr': 1e-5},
    #     {'params': head_params, 'lr': 1e-3}
    # ], weight_decay = 1e-4)

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = model.to(device)
    return criterion, device, model, optimizer


@app.cell
def _(criterion, device, model, optimizer, time, train_dataloader):
    model.train()

    num_epochs = 10
    epoch_plot = []
    loss_plot = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/30000}, Time:{time.time}")
        epoch_plot.append(epoch+1)
        loss_plot.append(running_loss/30000)
    return epoch_plot, loss_plot


@app.cell
def _(criterion, device, model, test_dataloader, torch):
    def evaluate_model():
        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for inputs, labels in test_dataloader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, predictions = torch.max(outputs, 1)

                correct += torch.sum(predictions == labels).item()
                total += labels.size(0)

        if total == 0:
            print("Warning: No test data found. Check your dataset or split logic.")
            return 0.0, 0.0

        test_loss = running_loss / total
        test_accuracy = correct / total

        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f} ({correct}/{total})')

        return test_loss, test_accuracy

    evaluate_model()
    return


@app.cell
def _(epoch_plot, loss_plot, plt):
    plt.plot(epoch_plot, loss_plot)
    plt.show()
    return


@app.cell
def _(GeoDataset, cell_id_dict, path, transform):
    ex_dataloader = GeoDataset(path=path, class_map = cell_id_dict, transform=transform, split = 'test', show_image = True)

    def _():
        for idx, i in enumerate(iter(ex_dataloader)):
            if idx == 14:
                return i


    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
