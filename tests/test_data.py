import torch
import os
from variables import PROJECT_PATH
import pytest


@pytest.mark.skipif(
    not os.path.exists("data/processed/"), reason="Data files not found"
)
def test_data():
    # Load data
    train_data = torch.load(
        os.path.join(str(PROJECT_PATH), "data/processed/traindata.pt")
    )
    test_data = torch.load(
        os.path.join(str(PROJECT_PATH), "data/processed/testdata.pt")
    )

    N_train = 25000
    N_test = 5000

    # Perform the data length tests
    assert len(train_data) == N_train, "Train data does not have the correct dim"
    assert len(test_data) == N_test, "Test data does not have the correct dim"

    # Check that each datapoint has shape [1,28,28]
    assert list(train_data[:][0][0].size()) == [
        28,
        28,
    ], "The sample must have dimension [1,28,28]"

    assert list(test_data[:][0][0].size()) == [
        28,
        28,
    ], "The sample must have dimension [1,28,28]"

    # Check that all labels are represented in data
    assert all(
        i in train_data[:][1] for i in range(10)
    ), "Not all labels are represented in train data"
    assert all(
        i in test_data[:][1] for i in range(10)
    ), "Not all labels are represented in test data"


train_data = torch.load(os.path.join(str(PROJECT_PATH), "data/processed/traindata.pt"))
test_data = torch.load(os.path.join(str(PROJECT_PATH), "data/processed/testdata.pt"))


@pytest.mark.skipif(
    not os.path.exists("data/processed/"), reason="Data files not found"
)
@pytest.mark.parametrize(
    "test_input,expected", [("len(train_data)", 25000), ("len(test_data)", 5000)]
)
def test_eval(test_input, expected):
    assert eval(test_input) == expected
