import pytest
from pathlib import Path

# The integration mark skips tests when the HCP-YA data mount is unavailable.
HCPYA_BASE = Path('/mnt/lindenmp_bignas/lindenmp/research_data/HCP_YA')

integration = pytest.mark.skipif(
    not HCPYA_BASE.exists(),
    reason="HCP-YA data mount not available",
)


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
