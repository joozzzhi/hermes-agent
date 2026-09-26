import pytest


@pytest.fixture(autouse=True)
def _isolate_recall_signal():
    """Состояние «контекст собирает движок» глобально на процесс — сбрасываем между тестами."""
    from plugins.memory.aida import recall_signal

    recall_signal.reset()
    yield
    recall_signal.reset()
