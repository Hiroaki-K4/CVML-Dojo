import torch


def create_tensors():
    # From data
    data = [[1, 2], [3, 4]]
    tensor_from_data = torch.tensor(data)

    # With specific shapes
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)
    eye = torch.eye(3)

    # Like another tensor
    similar = torch.ones_like(tensor_from_data)

    print("create_tensors")
    print(tensor_from_data)
    print(zeros)
    print(ones)
    print(rand)
    print(eye)
    print(similar)
    print()


def check_tensor_property():
    tensor = torch.rand(3, 4)
    print("check_tensor_property")
    print("Shape: ", tensor.shape)
    print("Datatype: ", tensor.dtype)
    print("Device: ", tensor.device)
    print()


def reshape_tensor():
    tensor = torch.arange(16)
    reshaped = tensor.view(4, 4)
    flattend = reshaped.flatten()
    print("reshape_tensor")
    print(reshaped)
    print(flattend)
    print()


def concat_and_stack():
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3)
    concat = torch.cat((tensor1, tensor2), dim=0)
    stack = torch.stack((tensor1, tensor2), dim=1)
    print("concat_and_stack")
    print(concat)
    print(stack.shape)
    print()


def mathematical_operations():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    # Element-wise operations
    print(x + y)
    print(x * y)
    print(x / y)
    print(torch.sqrt(x))
    # Matrix manipulation
    matrix1 = torch.rand(2, 3)
    matrix2 = torch.rand(3, 2)
    matmul = torch.mm(matrix1, matrix2)
    print(matmul)


if __name__ == "__main__":
    create_tensors()
    check_tensor_property()
    reshape_tensor()
    concat_and_stack()
    mathematical_operations()
