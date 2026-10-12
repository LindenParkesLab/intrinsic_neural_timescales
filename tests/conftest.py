import pytest
from pathlib import Path

_DATA_ROOT = Path('/mnt/lindenmp_bignas/lindenmp/research_data')

HCPYA_BASE   = _DATA_ROOT / 'HCP_YA'
HCPD_BASE    = _DATA_ROOT / 'HCP_D'
PENNLEAD_BASE = _DATA_ROOT / 'Penn_LEAD'
AOMIC_PIOP2_BASE = _DATA_ROOT / 'AOMIC_PIOP2'
AOMIC_PIOP1_BASE = _DATA_ROOT / 'AOMIC_PIOP1'
TCP_BASE     = _DATA_ROOT / 'TCP'

# Per-dataset integration marks — tests are auto-skipped when the mount is absent.
integration          = pytest.mark.skipif(not HCPYA_BASE.exists(),    reason="HCP-YA data mount not available")
integration_hcpd     = pytest.mark.skipif(not HCPD_BASE.exists(),     reason="HCP-D data mount not available")
integration_pennlead = pytest.mark.skipif(not PENNLEAD_BASE.exists(), reason="Penn_LEAD data mount not available")
integration_aomic    = pytest.mark.skipif(not AOMIC_PIOP2_BASE.exists(), reason="AOMIC-PIOP2 data mount not available")
integration_aomic_piop1 = pytest.mark.skipif(not AOMIC_PIOP1_BASE.exists(), reason="AOMIC-PIOP1 data mount not available")
integration_tcp      = pytest.mark.skipif(not TCP_BASE.exists(),      reason="TCP data mount not available")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires real data mounts to be accessible",
    )


@pytest.fixture(scope='session')
def hcpya_loader():
    """Session-scoped DataLoader for HCP-YA. Behavior data not loaded for speed."""
    from src.data import DataLoader
    return DataLoader('HCP-YA', load_behavior=False, verbose=False)


@pytest.fixture(scope='session')
def hcpd_loader():
    """Session-scoped DataLoader for HCP-D. Behavior data not loaded for speed."""
    from src.data import DataLoader
    return DataLoader('HCP-D', load_behavior=False, verbose=False)


@pytest.fixture(scope='session')
def penn_lead_loader():
    """Session-scoped DataLoader for Penn_LEAD. Behavior data not loaded for speed."""
    from src.data import DataLoader
    return DataLoader('Penn_LEAD', load_behavior=False, verbose=False)


@pytest.fixture(scope='session')
def aomic_piop2_loader():
    """Session-scoped DataLoader for AOMIC-PIOP2. Behavior data not loaded for speed."""
    from src.data import DataLoader
    return DataLoader('AOMIC_PIOP2', load_behavior=False, verbose=False)


@pytest.fixture(scope='session')
def aomic_piop1_loader():
    """Session-scoped DataLoader for AOMIC-PIOP1. Behavior data not loaded for speed."""
    from src.data import DataLoader
    return DataLoader('AOMIC_PIOP1', load_behavior=False, verbose=False)


@pytest.fixture(scope='session')
def tcp_loader():
    """Session-scoped DataLoader for TCP. Behavior data not loaded for speed."""
    from src.data import DataLoader
    return DataLoader('TCP', load_behavior=False, verbose=False)
