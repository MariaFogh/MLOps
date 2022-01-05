import argparse
import sys
import os
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim


def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train = torch.load("/data/processed/traindata.pt")
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    model.train()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    train_loss = []
    epochs = 30
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            # TODO: Training pass
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Training loss:" + str(running_loss / len(train_set)))

        train_loss.append(running_loss / len(train_set))

    os.makedirs("models/", exist_ok=True)
    torch.save(model, "models/trained_model.pt")

    plt.plot(train_loss)
    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/learningcurve.png")


if __name__ == "__main__":
    train()
