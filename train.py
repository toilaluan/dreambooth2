from data.T5FaceDataset import T5FaceDataset
from torch.utils.data import DataLoader
from transformers import Blip2Processor
from t5_face import T5Face
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = T5Face
dataset = T5FaceDataset("dataset/portraits/", processor)
print(dataset[0])

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)


for x, y in dataloader:
    print(x)
    print(y)
    break

EPOCH = 5
LEARNING_RATE = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCH):
    print(f"Epoch {epoch} started")
    for x, y in dataloader:

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
    print(f"Epoch {epoch} completed")
