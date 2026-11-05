import re
import sys

from robot import Robot
from ejecutor import Ejecutor

try:
    from detector_seguridad import DetectorSeguridad
except ImportError:
    DetectorSeguridad = None


class ClasificadorIntencion:
    TIPOS = (
        "MOVER",
        "GIRAR",
        "DETENERSE",
        "SALUDO",
        "CONSULTAR_ESTADO",
        "INSPECCIONAR_SEGURIDAD",
        "DESCONOCIDO",
    )

    def clasificar(self, texto):
        t = texto.lower()
        if any(p in t for p in (
            "inspeccion", "epp", "seguridad", "casco", "barbijo",
            "chaleco", "guante", "delantal", "auditiva", "cofia",
            "mascarilla", "tapaboca", "protecci", "escane",
            "medico", "médico", "industrial", "camara", "cámara"
        )):
            return "INSPECCIONAR_SEGURIDAD"
        if any(p in t for p in ("no avanc", "no te muevas", "detente", "para", "pará", "frena", "quieto")):
            return "DETENERSE"
        if any(p in t for p in ("retroced", "avanz", "mov", "muév", "muev", "adelante", "camin", "atrás", "atras")):
            return "MOVER"
        if any(p in t for p in ("gir", "rot", "vuelta", "dá una vuelta")):
            return "GIRAR"
        if any(p in t for p in ("salud", "hola")):
            return "SALUDO"
        if any(p in t for p in ("bater", "como estas", "cómo estás", "estado")):
            return "CONSULTAR_ESTADO"
        return "DESCONOCIDO"


class ExtractorParametros:
    def extraer(self, texto):
        params = {}
        t = texto.lower()

        m_dist = re.search(r'(\d+(?:\.\d+)?)\s*(?:metros|metro|m)(?!\s*/\s*s|\s*por\s*seg)\b', t)
        if m_dist:
            params["distancia_m"] = float(m_dist.group(1))

        m_ang = re.search(r'(\d+(?:\.\d+)?)\s*(?:grados|grado|°)', t)
        if m_ang:
            params["angulo_deg"] = float(m_ang.group(1))
        elif "media vuelta" in t:
            params["angulo_deg"] = 180
        elif "un cuarto de vuelta" in t:
            params["angulo_deg"] = 90
        elif "tres cuartos de vuelta" in t:
            params["angulo_deg"] = 270
        elif "vuelta completa" in t:
            params["angulo_deg"] = 360

        m_vel = re.search(r'(\d+(?:\.\d+)?)\s*m/s', t)
        if m_vel:
            params["velocidad_ms"] = float(m_vel.group(1))
        elif "despacio" in t or "lento" in t:
            params["velocidad_ms"] = 0.2
        elif "rápido" in t or "rapido" in t or "veloz" in t:
            params["velocidad_ms"] = 0.5

        if "derecha" in t:
            params["direccion"] = "derecha"
        elif "izquierda" in t:
            params["direccion"] = "izquierda"
        elif "atrás" in t or "atras" in t:
            params["direccion"] = "atras"

        if any(p in t for p in ("medico", "médico", "medica", "médica", "hospital", "cofia", "delantal")):
            params["modo_epp"] = "medico"
        elif any(p in t for p in ("industrial", "obra", "fabrica", "fábrica", "chaleco", "auditiva")):
            params["modo_epp"] = "industrial"
        elif "todos" in t or "ambos" in t:
            params["modo_epp"] = "ambos"

        return params


class AgenteRobot:
    def __init__(self, robot=None):
        self.robot = robot
        self.clasificador = ClasificadorIntencion()
        self.extractor = ExtractorParametros()
        self.ejecutor = Ejecutor(robot) if robot else None
        self.detector_seguridad = DetectorSeguridad() if DetectorSeguridad is not None else None
        self.modo_epp = "ambos"

    def reaccionar_seguridad(self, estado, detectados, infracciones):
        if self.robot is None:
            return

        try:
            if estado == "APROBADO":
                nombres = [x["clase"] for x in detectados]
                print(f"      [ROBOT] EPP OK ({nombres}) -> Saludando (Acceso permitido)")
                self.robot.saludar()
            elif estado == "RECHAZADO":
                faltantes = [x["clase"] for x in infracciones]
                print(f"      [ROBOT] Infracción ({faltantes}) -> Negando acceso (giro izq-der)")
                vel_giro = min(0.8, self.robot.perfil.velocidad_angular_max)
                self.robot.girar(velocidad=vel_giro, tiempo=0.5)
                self.robot.girar(velocidad=-vel_giro, tiempo=1.0)
                self.robot.girar(velocidad=vel_giro, tiempo=0.5)
                self.robot.detenerse()
        except Exception as exc:
            print(f"      [ROBOT] Error al ejecutar reacción: {exc}")

    def procesar(self, texto):
        tipo = self.clasificador.clasificar(texto)
        parametros = self.extractor.extraer(texto)

        if tipo == "DESCONOCIDO":
            return {
                "tipo": tipo,
                "parametros": parametros,
                "mensaje": "No entendí el comando.",
            }

        mensaje = f"Comando {tipo}"

        if tipo == "INSPECCIONAR_SEGURIDAD":
            if "modo_epp" in parametros:
                self.modo_epp = parametros["modo_epp"]

            if self.detector_seguridad and self.detector_seguridad.esta_listo():
                t = texto.lower()
                if "vivo" in t or "tiempo real" in t or "camara" in t or "cámara" in t:
                    self.detector_seguridad.abrir_camara_vivo(
                        modo=self.modo_epp,
                        callback_reaccion=self.reaccionar_seguridad,
                    )
                    mensaje = f"Inspección en vivo ({self.modo_epp}) finalizada."
                else:
                    res = self.detector_seguridad.inspeccionar_camara(
                        modo=self.modo_epp,
                        duracion_seg=3.0,
                    )
                    if res["exito"]:
                        elem = [x["clase"] for x in res["elementos_detectados"]]
                        infr = [x["clase"] for x in res["infracciones_detectadas"]]
                        mensaje = f"{res['mensaje']} | Detectados: {elem} | Infracciones: {infr}"
                        if infr:
                            self.reaccionar_seguridad("RECHAZADO", res["elementos_detectados"], res["infracciones_detectadas"])
                        elif elem:
                            self.reaccionar_seguridad("APROBADO", res["elementos_detectados"], res["infracciones_detectadas"])
                    else:
                        mensaje = res["mensaje"]
            else:
                mensaje = "Modelo YOLO no cargado o sin cámara conectada."
        elif self.ejecutor is not None:
            mensaje = self.ejecutor.ejecutar(tipo, parametros)

        return {
            "tipo": tipo,
            "parametros": parametros,
            "mensaje": mensaje,
        }


def main():
    sin_robot = "--sin-robot" in sys.argv

    robot = None
    if not sin_robot:
        robot = Robot()
        robot.conectar()

    try:
        agente = AgenteRobot(robot)
        print("\n  Comandos disponibles:")
        print("    - Movimientos: 'avanza 2 metros', 'gira 90 grados a la derecha', 'saluda', 'frena'")
        print("    - Inspección 3s: 'inspeccionar medico', 'inspeccionar industrial'")
        print("    - Cámara en vivo: 'camara en vivo medico', 'camara en vivo industrial'")
        print("  Enter vacío para salir.")
        while True:
            try:
                texto = input("\n  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not texto:
                break
            r = agente.procesar(texto)
            print(f"    [{r['tipo']}] {r.get('mensaje', '')}")
    finally:
        if robot is not None:
            robot.detenerse()
            robot.desconectar()


if __name__ == "__main__":
    main()
