import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
import pytest
import torch
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tensor():
    """Create a sample PyTorch tensor for testing."""
    return torch.randn(10, 3, 32, 32)


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return torch.randint(0, 10, (10,))


@pytest.fixture
def mock_dataset():
    """Create a mock PyTorch dataset."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(return_value=(torch.randn(3, 32, 32), torch.tensor(5)))
    return dataset


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = Mock()
    model.forward = Mock(return_value=torch.randn(1, 10))
    model.train = Mock()
    model.eval = Mock()
    return model


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'num_classes': 10,
        'device': 'cpu'
    }


@pytest.fixture
def sample_numpy_array():
    """Create a sample numpy array for testing."""
    return np.random.randn(100, 10)


@pytest.fixture
def sample_data_path(temp_dir):
    """Create a sample data file path in temporary directory."""
    data_file = temp_dir / "sample_data.pt"
    torch.save({'data': torch.randn(10, 3, 32, 32), 'labels': torch.randint(0, 10, (10,))}, data_file)
    return data_file


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = Mock()
    optimizer.step = Mock()
    optimizer.zero_grad = Mock()
    return optimizer


@pytest.fixture
def mock_loss_function():
    """Create a mock loss function."""
    loss_fn = Mock()
    loss_fn.return_value = torch.tensor(0.5, requires_grad=True)
    return loss_fn


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')