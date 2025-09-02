"""
Validation tests to ensure the testing infrastructure is working correctly.
"""
import pytest
import torch
import numpy as np


def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True


def test_fixtures_available(temp_dir, sample_tensor, sample_labels, sample_config):
    """Test that common fixtures are available and working."""
    assert temp_dir.exists()
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (10, 3, 32, 32)
    assert isinstance(sample_labels, torch.Tensor)
    assert sample_labels.shape == (10,)
    assert isinstance(sample_config, dict)
    assert 'batch_size' in sample_config


def test_torch_available():
    """Test that PyTorch is available and working."""
    tensor = torch.randn(2, 3)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float32


def test_numpy_available():
    """Test that NumPy is available and working."""
    array = np.random.randn(2, 3)
    assert array.shape == (2, 3)


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker is working."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker is working."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker is working."""
    import time
    time.sleep(0.01)  # Very short sleep for testing
    assert True


def test_mock_available(mock_dataset, mock_model):
    """Test that mock fixtures are available."""
    assert len(mock_dataset) == 100
    data, label = mock_dataset[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    
    output = mock_model.forward(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 10)


def test_temporary_files(temp_dir, sample_data_path):
    """Test that temporary file fixtures work."""
    assert sample_data_path.exists()
    data = torch.load(sample_data_path)
    assert 'data' in data
    assert 'labels' in data