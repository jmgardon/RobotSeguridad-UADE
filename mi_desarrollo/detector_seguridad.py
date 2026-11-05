from collections import Counter
import csv
from datetime import datetime
from pathlib import Path
import threading
import time
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_DISPONIBLE = True
except ImportError:
    ULTRALYTICS_DISPONIBLE = False


class DetectorSeguridad:
    PERFILES = {
        "medico": {
            "ok": {
                "DelantalMedico",
                "barbijoMedico",
                "cofiaMedico",
                "guantesMedicos",
            },
            "infracciones": {
                "noBarbijo",
                "noCofia",
                "noCofianoBarbijo",
                "noDelantalMedico",
                "noGuantes",
            },
        },
        "industrial": {
            "ok": {
                "barbijoIndustrial",
                "chalecoSeguridad",
                "guantesIndustriales",
                "proteccionAuditiva",
            },
            "infracciones": {
                "noBarbijo",
                "noBarbijoNoProteccionAudictiva",
                "noChalecoSeguridad",
                "noGuantes",
                "noProteccionAuditiva",
            },
        },
    }

    ELEMENTOS_OK = PERFILES["medico"]["ok"] | PERFILES["industrial"]["ok"]
    INFRACCIONES = PERFILES["medico"]["infracciones"] | PERFILES["industrial"]["infracciones"]

    # Umbral más sensible para barbijos y guantes (objetos chicos en cámara)
    CLASES_SENSIBLES = {
        "barbijoMedico",
        "barbijoIndustrial",
        "noBarbijo",
        "guantesMedicos",
        "guantesIndustriales",
        "noGuantes",
    }
    UMBRAL_CLASE_SENSIBLE = 0.25
    UMBRAL_GENERAL = 0.38

    def __init__(self, ruta_modelo=None):
        self.base_dir = Path(__file__).resolve().parent
        if ruta_modelo is None:
            ruta_modelo = self.base_dir / "train_v2" / "weights" / "best.pt"

        self.ruta_modelo = Path(ruta_modelo)
        self.ruta_csv = self.base_dir / "registro_accesos.csv"
        self.carpeta_evidencias = self.base_dir / "evidencias"
        self.modelo = None
        self._mapa_nombre_a_id = {}

        if ULTRALYTICS_DISPONIBLE and self.ruta_modelo.exists():
            try:
                self.modelo = YOLO(str(self.ruta_modelo))
                self._mapa_nombre_a_id = {
                    nombre: idx for idx, nombre in self.modelo.names.items()
                }
            except Exception as e:
                print(f"[DetectorSeguridad] Error cargando modelo: {e}")
        else:
            if not ULTRALYTICS_DISPONIBLE:
                print("[DetectorSeguridad] ADVERTENCIA: 'ultralytics' no está instalado.")
            elif not self.ruta_modelo.exists():
                print(f"[DetectorSeguridad] ADVERTENCIA: No se encontró el modelo en {self.ruta_modelo}")

    def esta_listo(self):
        return self.modelo is not None

    def registrar_evento(self, modo: str, resultado: str, detectados: list, infracciones: list, frame_anotado=None):
        ahora = datetime.now()
        fecha_hora_str = ahora.strftime("%Y-%m-%d %H:%M:%S")
        elem_str = "; ".join(f"{x['clase']} ({x['confianza']})" for x in detectados) or "ninguno"
        infr_str = "; ".join(f"{x['clase']} ({x['confianza']})" for x in infracciones) or "ninguna"

        archivo_nuevo = not self.ruta_csv.exists()
        try:
            with open(self.ruta_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if archivo_nuevo:
                    writer.writerow([
                        "fecha_hora",
                        "modo",
                        "resultado",
                        "elementos_detectados",
                        "infracciones_detectadas",
                    ])
                writer.writerow([fecha_hora_str, modo.upper(), resultado, elem_str, infr_str])
        except Exception as e:
            print(f"[DetectorSeguridad] No se pudo escribir en registro_accesos.csv: {e}")

        if frame_anotado is not None:
            try:
                self.carpeta_evidencias.mkdir(parents=True, exist_ok=True)
                nombre_foto = f"{ahora.strftime('%Y%m%d_%H%M%S')}_{resultado}_{modo.lower()}.jpg"
                cv2.imwrite(str(self.carpeta_evidencias / nombre_foto), frame_anotado)
            except Exception as e:
                print(f"[DetectorSeguridad] No se pudo guardar foto de evidencia: {e}")

    def _ids_para_modo(self, modo: str):
        modo = (modo or "ambos").lower()
        if modo not in self.PERFILES:
            return None
        # Permitir ambos tipos de barbijo y guantes en la inferencia por si la luz
        # hace que un barbijoMedico se vea como barbijoIndustrial (o viceversa)
        clases_compartidas = {
            "barbijoMedico",
            "barbijoIndustrial",
            "guantesMedicos",
            "guantesIndustriales",
        }
        clases_modo = (
            self.PERFILES[modo]["ok"]
            | self.PERFILES[modo]["infracciones"]
            | clases_compartidas
        )
        ids = [self._mapa_nombre_a_id[c] for c in clases_modo if c in self._mapa_nombre_a_id]
        return ids if ids else None

    def _es_infraccion(self, nombre_clase: str) -> bool:
        return nombre_clase in self.INFRACCIONES or nombre_clase.lower().startswith("no")

    def _inferir_frame(self, frame, modo_actual):
        """Corre YOLO con umbral bajo para no perder guantes y filtra por clase."""
        ids_filtro = self._ids_para_modo(modo_actual)
        results = self.modelo.predict(
            frame,
            conf=self.UMBRAL_CLASE_SENSIBLE,
            iou=0.45,
            classes=ids_filtro,
            verbose=False,
        )
        res = results[0]
        canvas = frame.copy()
        detecciones = []

        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            nombre_clase = self.modelo.names[cls_id]

            umbral_min = (
                self.UMBRAL_CLASE_SENSIBLE
                if nombre_clase in self.CLASES_SENSIBLES
                else self.UMBRAL_GENERAL
            )
            if conf_val < umbral_min:
                continue

            detecciones.append((nombre_clase, conf_val))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            es_inf = self._es_infraccion(nombre_clase)
            color = (0, 0, 230) if es_inf else (0, 200, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"{nombre_clase} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_top = max(y1, th + 8)
            cv2.rectangle(canvas, (x1, y_top - th - 8), (x1 + tw + 8, y_top), color, -1)
            cv2.putText(
                canvas,
                etiqueta,
                (x1 + 4, y_top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return canvas, detecciones

    def _estabilizar_ventana(self, historial_frames, umbral_presencia=0.30):
        if not historial_frames:
            return [], []

        total_frames = len(historial_frames)
        conteo_clases = Counter()
        conf_acumulada = Counter()

        for detecciones_frame in historial_frames:
            vistas_en_frame = {}
            for nombre_clase, conf in detecciones_frame:
                if nombre_clase not in vistas_en_frame or conf > vistas_en_frame[nombre_clase]:
                    vistas_en_frame[nombre_clase] = conf
            for nombre_clase, conf in vistas_en_frame.items():
                conteo_clases[nombre_clase] += 1
                conf_acumulada[nombre_clase] += conf

        detectados = []
        infracciones = []

        for nombre_clase, apariciones in conteo_clases.items():
            # Guantes requieren menos frames porque las manos se mueven más
            factor = 0.22 if nombre_clase in self.CLASES_SENSIBLES else umbral_presencia
            min_apariciones = max(2, int(total_frames * factor))
            if apariciones >= min_apariciones:
                conf_media = round(conf_acumulada[nombre_clase] / apariciones, 2)
                item = {"clase": nombre_clase, "confianza": conf_media}
                if self._es_infraccion(nombre_clase):
                    infracciones.append(item)
                else:
                    detectados.append(item)

        # Si en el mismo escaneo detectó guantes o barbijo puestos, descartar falso negativo
        clases_ok = {x["clase"] for x in detectados}
        if ("guantesMedicos" in clases_ok or "guantesIndustriales" in clases_ok):
            if conteo_clases.get("guantesMedicos", 0) + conteo_clases.get("guantesIndustriales", 0) >= conteo_clases.get("noGuantes", 0):
                infracciones = [x for x in infracciones if x["clase"] != "noGuantes"]

        if ("barbijoMedico" in clases_ok or "barbijoIndustrial" in clases_ok):
            if conteo_clases.get("barbijoMedico", 0) + conteo_clases.get("barbijoIndustrial", 0) >= conteo_clases.get("noBarbijo", 0):
                infracciones = [x for x in infracciones if x["clase"] != "noBarbijo"]

        return detectados, infracciones

    def _dibujar_efecto_scanner(self, frame, progreso: float):
        """Dibuja una línea láser de escaneo que recorre la pantalla."""
        alto, ancho = frame.shape[:2]
        # El láser baja y sube durante el escaneo
        ciclo = (progreso * 2.0) % 2.0
        pos_rel = ciclo if ciclo <= 1.0 else (2.0 - ciclo)
        y_laser = int(75 + pos_rel * (alto - 90))

        overlay = frame.copy()
        y_banda_sup = max(70, y_laser - 25)
        y_banda_inf = min(alto, y_laser + 25)
        cv2.rectangle(overlay, (0, y_banda_sup), (ancho, y_banda_inf), (255, 220, 0), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.line(frame, (0, y_laser), (ancho, y_laser), (255, 255, 100), 3)

        # Barra de progreso arriba
        ancho_barra = int((ancho - 24) * min(1.0, max(0.0, progreso)))
        cv2.rectangle(frame, (12, 62), (ancho - 12, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (12, 62), (12 + ancho_barra, 70), (0, 255, 255), -1)
        return frame

    def _dibujar_hud_scanner(self, frame, modo, fase, texto_principal, subtitulo, color_fondo):
        alto, ancho = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (ancho, 74), color_fondo, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        titulo = f"MODO: {modo.upper()} | {texto_principal}"
        cv2.putText(frame, titulo, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, subtitulo, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
        return frame

    def abrir_camara_vivo(
        self,
        source=0,
        conf=0.50,
        modo="ambos",
        callback_reaccion=None,
        duracion_scan_seg=2.8,
        intervalo_auto_seg=10.0,
    ):
        if not self.esta_listo():
            print("[DetectorSeguridad] No se puede iniciar: modelo no disponible.")
            return

        modo_actual = (modo or "ambos").lower()
        print(f"[DetectorSeguridad] Escáner en vivo iniciado (modo={modo_actual}).")
        print("  Controles en la ventana:")
        print("    [ESPACIO] o [E] -> Iniciar escaneo ahora")
        print("    [A]             -> Activar/Desactivar Auto-Scanner periódico")
        print("    [M] Médico | [I] Industrial | [T] Todos | [Q] Salir")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[DetectorSeguridad] Error: No se pudo acceder a la cámara en el índice {source}.")
            return

        # Estados del escáner: "ESPERA", "ESCANEANDO", "RESULTADO"
        fase = "ESPERA"
        auto_scan = False
        inicio_fase = time.monotonic()
        ultimo_fin_resultado = time.monotonic()
        frames_escaneo = []
        resultado_actual = None
        det_resultado = []
        inf_resultado = []
        robot_ocupado = threading.Lock()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DetectorSeguridad] Fin del stream de cámara.")
                break

            ahora = time.monotonic()
            annotated_frame, detecciones_actuales = self._inferir_frame(frame, modo_actual)

            auto_txt = "ON" if auto_scan else "OFF"
            controles_txt = f"[ESPACIO] Escanear | [A] Auto:{auto_txt} | [M] Medico [I] Ind [T] Todos | [Q] Salir"

            if fase == "ESPERA":
                if auto_scan:
                    restante_auto = max(0.0, intervalo_auto_seg - (ahora - ultimo_fin_resultado))
                    texto_top = f"LISTO - Auto-Scan en {restante_auto:.1f}s"
                    if restante_auto <= 0.0:
                        fase = "ESCANEANDO"
                        inicio_fase = ahora
                        frames_escaneo.clear()
                else:
                    texto_top = "LISTO PARA ESCANEAR (Presiona ESPACIO)"

                annotated_frame = self._dibujar_hud_scanner(
                    annotated_frame,
                    modo_actual,
                    fase,
                    texto_top,
                    controles_txt,
                    color_fondo=(65, 65, 65),
                )

            elif fase == "ESCANEANDO":
                frames_escaneo.append(detecciones_actuales)
                transcurrido = ahora - inicio_fase
                progreso = transcurrido / duracion_scan_seg
                porcentaje = min(100, int(progreso * 100))

                annotated_frame = self._dibujar_hud_scanner(
                    annotated_frame,
                    modo_actual,
                    fase,
                    f"ESCANEANDO PERSONA... {porcentaje}%",
                    "Mantene tu posicion frente a la camara...",
                    color_fondo=(160, 110, 0),
                )
                annotated_frame = self._dibujar_efecto_scanner(annotated_frame, progreso)

                if transcurrido >= duracion_scan_seg:
                    det_resultado, inf_resultado = self._estabilizar_ventana(frames_escaneo)
                    if inf_resultado:
                        resultado_actual = "RECHAZADO"
                    elif det_resultado:
                        resultado_actual = "APROBADO"
                    else:
                        resultado_actual = "SIN_DETECCIONES"

                    fase = "RESULTADO"
                    inicio_fase = ahora

                    if resultado_actual in ("APROBADO", "RECHAZADO"):
                        copia_evidencia = annotated_frame.copy()
                        modo_ev = modo_actual
                        res_ev = resultado_actual
                        det_ev = list(det_resultado)
                        inf_ev = list(inf_resultado)

                        def _disparar(est, det, inf, frm, md):
                            with robot_ocupado:
                                self.registrar_evento(md, est, det, inf, frame_anotado=frm)
                                if callback_reaccion is not None:
                                    callback_reaccion(est, det, inf)

                        if not robot_ocupado.locked():
                            threading.Thread(
                                target=_disparar,
                                args=(res_ev, det_ev, inf_ev, copia_evidencia, modo_ev),
                                daemon=True,
                            ).start()

            elif fase == "RESULTADO":
                transcurrido = ahora - inicio_fase
                if resultado_actual == "APROBADO":
                    nombres_ok = ", ".join(x["clase"] for x in det_resultado[:4])
                    texto_top = f"ACCESO PERMITIDO ({nombres_ok})"
                    color_res = (0, 155, 0)
                elif resultado_actual == "RECHAZADO":
                    nombres_inf = ", ".join(x["clase"] for x in inf_resultado[:4])
                    texto_top = f"ACCESO DENEGADO - FALTA: {nombres_inf}"
                    color_res = (0, 0, 190)
                else:
                    texto_top = "SIN DETECCIONES CLARAS - Intenta de nuevo"
                    color_res = (90, 90, 90)

                annotated_frame = self._dibujar_hud_scanner(
                    annotated_frame,
                    modo_actual,
                    fase,
                    texto_top,
                    controles_txt,
                    color_fondo=color_res,
                )

                if transcurrido >= 3.5:
                    fase = "ESPERA"
                    ultimo_fin_resultado = ahora

            cv2.imshow("Robot AI - Escaner de Seguridad EPP", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord(" "), ord("e")) and fase != "ESCANEANDO":
                fase = "ESCANEANDO"
                inicio_fase = time.monotonic()
                frames_escaneo.clear()
                print(f"[DetectorSeguridad] Escaneo iniciado ({modo_actual.upper()})...")
            elif key == ord("a"):
                auto_scan = not auto_scan
                ultimo_fin_resultado = time.monotonic()
                print(f"[DetectorSeguridad] Auto-Scanner: {'ACTIVADO' if auto_scan else 'DESACTIVADO'}")
            elif key == ord("m"):
                modo_actual = "medico"
                fase = "ESPERA"
                print("[DetectorSeguridad] Cambiado a MODO MÉDICO.")
            elif key == ord("i"):
                modo_actual = "industrial"
                fase = "ESPERA"
                print("[DetectorSeguridad] Cambiado a MODO INDUSTRIAL.")
            elif key == ord("t"):
                modo_actual = "ambos"
                fase = "ESPERA"
                print("[DetectorSeguridad] Cambiado a MODO TODOS.")

        cap.release()
        cv2.destroyAllWindows()

    def inspeccionar_camara(self, source=0, conf=0.50, mostrar=True, modo="ambos", duracion_seg=2.8):
        if not self.esta_listo():
            return {
                "exito": False,
                "modo": modo,
                "mensaje": "Modelo YOLO no cargado o ultralytics no instalado.",
                "elementos_detectados": [],
                "infracciones_detectadas": [],
            }

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {
                "exito": False,
                "modo": modo,
                "mensaje": f"No se pudo conectar a la cámara (índice {source}).",
                "elementos_detectados": [],
                "infracciones_detectadas": [],
            }

        for _ in range(5):
            cap.read()

        modo_actual = (modo or "ambos").lower()
        ventana_frames = []
        ultimo_frame_anotado = None
        inicio = time.monotonic()

        while (time.monotonic() - inicio) < duracion_seg:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            transcurrido = time.monotonic() - inicio
            progreso = transcurrido / duracion_seg
            porcentaje = min(100, int(progreso * 100))

            annotated_frame, detecciones_actuales = self._inferir_frame(frame, modo_actual)
            ventana_frames.append(detecciones_actuales)

            annotated_frame = self._dibujar_hud_scanner(
                annotated_frame,
                modo_actual,
                "ESCANEANDO",
                f"ESCANEANDO PERSONA... {porcentaje}%",
                "Mantene tu posicion frente a la camara...",
                color_fondo=(160, 110, 0),
            )
            ultimo_frame_anotado = self._dibujar_efecto_scanner(annotated_frame, progreso)

            if mostrar:
                cv2.imshow("Robot AI - Escaner de Seguridad EPP", ultimo_frame_anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if mostrar:
            cv2.destroyAllWindows()

        if not ventana_frames:
            return {
                "exito": False,
                "modo": modo_actual,
                "mensaje": "No se pudo obtener imagen de la cámara.",
                "elementos_detectados": [],
                "infracciones_detectadas": [],
            }

        detectados, infracciones = self._estabilizar_ventana(ventana_frames)
        cumple_seguridad = (len(infracciones) == 0) and (len(detectados) > 0)

        if len(infracciones) > 0:
            resultado_str = "RECHAZADO"
            estado_msg = f"ACCESO DENEGADO ({modo_actual.upper()}): Se detectaron infracciones"
        elif len(detectados) > 0:
            resultado_str = "APROBADO"
            estado_msg = f"ACCESO PERMITIDO ({modo_actual.upper()}): Equipo de seguridad OK"
        else:
            resultado_str = "SIN_DETECCIONES"
            estado_msg = f"SIN DETECCIONES ({modo_actual.upper()}): No se detectaron elementos ni infracciones"

        if resultado_str in ("APROBADO", "RECHAZADO"):
            self.registrar_evento(
                modo_actual,
                resultado_str,
                detectados,
                infracciones,
                frame_anotado=ultimo_frame_anotado,
            )

        return {
            "exito": True,
            "modo": modo_actual,
            "cumple_seguridad": cumple_seguridad,
            "elementos_detectados": detectados,
            "infracciones_detectadas": infracciones,
            "mensaje": estado_msg,
        }


if __name__ == "__main__":
    detector = DetectorSeguridad()
    if detector.esta_listo():
        print("Modelo cargado correctamente. Iniciando escáner en vivo...")
        detector.abrir_camara_vivo()
    else:
        print("No se pudo cargar el modelo.")
