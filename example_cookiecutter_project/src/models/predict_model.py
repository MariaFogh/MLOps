import argparse
import sys

import torch
from torch import optim


def predict():
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("load_model_from", default="")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    # TODO: Implement evaluation logic here
    model = torch.load(args.load_model_from)
    test = torch.load("/data/processed/testdata.pt")
    test_set = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    with torch.no_grad():
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

    accuracy = torch.mean(equals.type(torch.FloatTensor))

    print("Accuracy:" + str(accuracy.item() * 100) + "%")


if __name__ == "__main__":
    predict()
