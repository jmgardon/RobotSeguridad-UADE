import sys
from pathlib import Path

_ENTORNO = Path(__file__).resolve().parent.parent / "entorno"
if str(_ENTORNO) not in sys.path:
    sys.path.insert(0, str(_ENTORNO))

from sim.robot import EstadoRobot, NoHaySimulador, Robot  # noqa: E402,F401
from sim.safety import ErrorDeSeguridad  # noqa: E402,F401

__all__ = ["Robot", "EstadoRobot", "ErrorDeSeguridad", "NoHaySimulador"]
