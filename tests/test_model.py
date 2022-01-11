import pytest
import torch
from src.models.model import MyAwesomeModel

# Test that a given input with shape X outputs shape Y using the model
def test_model():
    in_dim = 784
    h1_dim = 256
    h2_dim = 128
    h3_dim = 64
    out_dim = 10
    dropout = 0.2

    model = MyAwesomeModel(in_dim, h1_dim, h2_dim, h3_dim, out_dim, dropout)

    in_tensor = torch.rand(1, 784)
    out_tensor = model(in_tensor)

    assert list(out_tensor.size()) == [1, 10], "The output size must be [1,10]"


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected 2nd input to have shape 784"):
        input_dim = 784
        hidden_dim1 = 256
        hidden_dim2 = 128
        hidden_dim3 = 64
        output_dim = 10
        dropout = 0.2

        model = MyAwesomeModel(
            input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout
        )

        input_tensor = torch.rand(1, 123)
        model(input_tensor)
