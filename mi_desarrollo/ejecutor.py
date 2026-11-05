import math
import sys
from pathlib import Path

_ENTORNO = Path(__file__).resolve().parent.parent / "entorno"
if str(_ENTORNO) not in sys.path:
    sys.path.insert(0, str(_ENTORNO))

try:
    from sim.safety import ErrorDeSeguridad
except ImportError:
    class ErrorDeSeguridad(ValueError):
        pass


def partir_en_tramos(tiempo_total, tope):
    if tiempo_total <= tope:
        return [tiempo_total]
    tramos = []
    restante = float(tiempo_total)
    while restante > tope + 1e-9:
        tramos.append(tope)
        restante -= tope
    if restante > 1e-9:
        tramos.append(restante)
    return tramos


class Ejecutor:
    def __init__(self, robot, mostrar=True):
        self.robot = robot
        self.mostrar = mostrar
        self.perfil = robot.perfil

    def ejecutar(self, tipo, parametros):
        parametros = parametros or {}
        try:
            if tipo == "MOVER":
                return self._mover(parametros)
            if tipo == "GIRAR":
                return self._girar(parametros)
            if tipo == "DETENERSE":
                self.robot.detenerse()
                return "detenido"
            if tipo == "SALUDO":
                self.robot.saludar()
                return "saludo"
            if tipo == "CONSULTAR_ESTADO":
                e = self.robot.verificar_estado()
                return f"estado: {e}"
        except ErrorDeSeguridad as exc:
            return f"RECHAZADO POR EL ROBOT: {exc}"
        return f"no se como ejecutar {tipo}"

    def _mover(self, p):
        distancia = float(p.get("distancia_m", 0.5))
        velocidad = float(p.get("velocidad_ms", self.perfil.velocidad_max))
        velocidad = min(abs(velocidad), self.perfil.velocidad_max)
        if velocidad <= 0:
            return "velocidad cero: no hay nada que ejecutar"

        signo = -1.0 if p.get("direccion") == "atras" else 1.0
        tiempo_total = abs(distancia) / velocidad
        tramos = partir_en_tramos(tiempo_total, self.perfil.duracion_max)

        self._log(f"{distancia:g} m a {velocidad:g} m/s = {tiempo_total:.2f} s")
        for i, t in enumerate(tramos, 1):
            self._log(f"  tramo {i}/{len(tramos)}: avanzar({signo * velocidad:+.2f}, {t:.2f})")
            self.robot.avanzar(velocidad=signo * velocidad, tiempo=t)
        return f"avanzo {distancia:g} m en {len(tramos)} orden(es)"

    def _girar(self, p):
        grados = float(p.get("angulo_deg", 90))
        velocidad = float(p.get("velocidad_giro", self.perfil.velocidad_angular_max))
        velocidad = min(abs(velocidad), self.perfil.velocidad_angular_max)
        if velocidad <= 0:
            return "velocidad de giro cero"

        signo = -1.0 if p.get("direccion") == "derecha" else 1.0
        radianes = math.radians(abs(grados))
        tiempo_total = radianes / velocidad
        tramos = partir_en_tramos(tiempo_total, self.perfil.duracion_max)

        lado = "derecha" if signo < 0 else "izquierda"
        self._log(f"{grados:g} grados a la {lado} ({tiempo_total:.2f} s)")
        for i, t in enumerate(tramos, 1):
            self._log(f"  tramo {i}/{len(tramos)}: girar({signo * velocidad:+.2f}, {t:.2f})")
            self.robot.girar(velocidad=signo * velocidad, tiempo=t)
        return f"giro {grados:g} grados a la {lado} en {len(tramos)} orden(es)"

    def _log(self, texto):
        if self.mostrar:
            print(f"      [EJECUTOR] {texto}")
