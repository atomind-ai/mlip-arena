import pytest
from unittest.mock import MagicMock, patch

from ase.calculators.calculator import Calculator, BaseCalculator
from mlip_arena.models import MLIPEnum
from mlip_arena.models.utils import get_freer_device as models_get_freer_device
from mlip_arena.tasks.utils import get_freer_device as tasks_get_freer_device, get_calculator, _calculator_key_fn


class MockCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, *args, name="MockCalculator", **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self._name


def test_get_freer_device_cuda():
    """Test get_freer_device when CUDA GPUs are available."""
    with (
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.get_device_properties") as mock_properties,
        patch("torch.cuda.memory_allocated", return_value=100),
    ):
        # Mocking GPU properties
        gpu0 = MagicMock()
        gpu0.total_memory = 1000
        gpu1 = MagicMock()
        gpu1.total_memory = 2000
        mock_properties.side_effect = [gpu0, gpu1]

        # gpu1 has more free memory (2000-100 = 1900 vs 1000-100 = 900)
        device = models_get_freer_device()
        assert device.type == "cuda"
        assert device.index == 1

        # Do the same for the tasks module implementation
        mock_properties.side_effect = [gpu0, gpu1]
        device2 = tasks_get_freer_device()
        assert device2.type == "cuda"
        assert device2.index == 1


def test_get_freer_device_mps():
    """Test get_freer_device fallback to MPS when CUDA is not available."""
    with patch("torch.cuda.device_count", return_value=0), patch("torch.backends.mps.is_available", return_value=True):
        device = models_get_freer_device()
        assert device.type == "mps"

        device2 = tasks_get_freer_device()
        assert device2.type == "mps"


def test_get_freer_device_cpu():
    """Test get_freer_device fallback to CPU when neither CUDA nor MPS is available."""
    with patch("torch.cuda.device_count", return_value=0), patch("torch.backends.mps.is_available", return_value=False):
        device = models_get_freer_device()
        assert device.type == "cpu"

        device2 = tasks_get_freer_device()
        assert device2.type == "cpu"


def test_get_calculator_by_enum_and_string():
    """Test get_calculator using MLIPEnum and string inputs."""
    # Find any available member in MLIPEnum
    members = list(MLIPEnum)
    if not members:
        pytest.skip("No models registered in MLIPEnum")

    enum_member = members[0]
    string_name = enum_member.name

    # Mock the .load() method of the enum member to return a MockCalculator
    mock_calc = MockCalculator()
    with patch.object(enum_member, "load", return_value=mock_calc) as mock_load:
        # Test loading by MLIPEnum member
        res1 = get_calculator(enum_member, device="cpu")
        assert res1 == mock_calc
        mock_load.assert_called_once_with(device="cpu")
        assert res1.__str__() == enum_member.name

    # Mock the load method for string lookup
    with patch.object(MLIPEnum[string_name], "load", return_value=mock_calc) as mock_load_str:
        # Test loading by string name
        res2 = get_calculator(string_name, device="cpu")
        assert res2 == mock_calc
        mock_load_str.assert_called_once_with(device="cpu")
        assert res2.__str__() == string_name


def test_get_calculator_by_class_and_object():
    """Test get_calculator using class type and instance inputs."""
    # Test passing a Calculator class type
    res_cls = get_calculator(MockCalculator, device="cpu")
    assert isinstance(res_cls, MockCalculator)
    assert res_cls.__str__() == "MockCalculator"

    # Test passing a Calculator instance
    inst = MockCalculator()
    res_inst = get_calculator(inst, device="cpu")
    assert res_inst == inst
    assert res_inst.__str__() == "MockCalculator"


def test_get_calculator_invalid():
    """Test get_calculator with invalid input types raises ValueError."""
    with pytest.raises(ValueError, match="Invalid calculator:"):
        get_calculator(12345)


def test_get_calculator_dispersion():
    """Test get_calculator with dispersion=True."""
    # Mock TorchDFTD3Calculator
    mock_dftd_calc = MagicMock(spec=BaseCalculator)

    # We want to patch the import/class inside get_calculator
    with patch("torch_dftd.torch_dftd3_calculator.TorchDFTD3Calculator", return_value=mock_dftd_calc):
        res = get_calculator(MockCalculator, dispersion=True, device="cpu")
        # SumCalculator combines MockCalculator and TorchDFTD3Calculator
        from ase.calculators.mixing import SumCalculator

        assert isinstance(res, SumCalculator)


def test_get_calculator_dispersion_import_error():
    """Test get_calculator raises ImportError if torch_dftd is not installed."""
    with patch("builtins.__import__", side_effect=ImportError):
        with pytest.raises(ImportError, match="torch_dftd is required for dispersion"):
            get_calculator(MockCalculator, dispersion=True, device="cpu")


def test_calculator_key_fn():
    """Test _calculator_key_fn helper function."""
    mock_calc = MockCalculator(name="MyMockCalc")

    res = _calculator_key_fn(None, {"calculator": mock_calc})
    assert res == {"calculator": "MyMockCalc"}
