import torch
from wirehead import MongoheadDataset, MongoTupleheadDataset

def check_tensor_shape(tensor, expected_shape, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} is not a torch.Tensor. Got {type(tensor)}")
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} has incorrect shape. Expected {expected_shape}, got {tensor.shape}")

def test_datasets(expected_shape=(256, 256, 256)):
    # Test MongoheadDataset
    dataset = MongoheadDataset(config_path="config.yaml")
    idx = [0]
    data = dataset[idx]
    sample, label = data[0]['input'], data[0]['label']
    
    check_tensor_shape(sample, expected_shape, "MongoheadDataset sample")
    check_tensor_shape(label, expected_shape, "MongoheadDataset label")

    # Test MongoTupleheadDataset
    dataset = MongoTupleheadDataset(config_path="config.yaml")
    idx = [0]
    data = dataset[idx][0]
    sample, label = data[0], data[1]
    
    check_tensor_shape(sample, expected_shape, "MongoTupleheadDataset sample")
    check_tensor_shape(label, expected_shape, "MongoTupleheadDataset label")

    print("All tests passed successfully!")

if __name__ == "__main__":
    try:
        test_datasets()
    except Exception as e:
        print(f"Test failed: {str(e)}")
