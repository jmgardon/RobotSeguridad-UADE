# Robot Inspector de Seguridad (EPP) — UADE Inteligencia Artificial

Proyecto de **Inteligencia Artificial y Robótica** que integra el simulador oficial **Unitree (G1 / Go2)** en MuJoCo con un sistema de **Visión por Computadora (YOLOv8)** en tiempo real para el control de acceso y detección de **Elementos de Protección Personal (EPP)** en entornos **Médicos** e **Industriales**.

---

## Arquitectura del Agente (`Percepción -> Decisión -> Acción`)

El sistema funciona como un **Agente Inteligente** dividido en tres etapas:

1. **Percepción (`detector_seguridad.py` + `ClasificadorIntencion`):**
   * **Lenguaje Natural:** Interpreta órdenes escritas por consola en español rioplatense (movimientos, giros, saludos, estado de batería y comandos de inspección por cámara).
   * **Visión Artificial (YOLOv8s):** Detecta en tiempo real 16 clases entre equipos de protección válidos e infracciones/faltantes.
2. **Decisión (Filtrado Contextual + Estabilización Temporal):**
   * **Perfiles de Contexto (`Médico` vs. `Industrial`):** Utilizando un único modelo entrenado, el agente filtra dinámicamente las clases evaluadas según el área seleccionada para evitar falsas infracciones cruzadas (por ejemplo, no exigir chaleco reflectivo a un médico ni cofia a un operario industrial).
   * **Modo Escáner con Ventana de Estabilización:** En lugar de reaccionar a cuadros sueltos o parpadeos de la cámara, realiza un barrido de escaneo (~2.8 segundos) y confirma únicamente los elementos e infracciones con presencia estable.
   * **Umbrales de Confianza Diferenciados:** Aplica un umbral más sensible (`0.25`) para objetos pequeños/móviles en cámara (barbijos y guantes) y un umbral general (`0.38`) para prendas grandes (delantal, chaleco, cofia, protección auditiva).
3. **Acción (`tpRobot.py` + `ejecutor.py`):**
   * **Acceso Permitido (`APROBADO`):** Si la persona cuenta con elementos de seguridad y ninguna infracción, el robot **levanta el brazo y saluda** (`robot.saludar()`).
   * **Acceso Denegado (`RECHAZADO`):** Si se detecta alguna infracción/faltante, el robot realiza un **gesto físico de negación** girando de izquierda a derecha y deteniéndose.
   * **Auditoría Automática:** Cada inspección confirmada guarda una fila en `mi_desarrollo/registro_accesos.csv` (fecha/hora, modo, resultado, elementos e infracciones) y una captura de evidencia anotada en `mi_desarrollo/evidencias/`.

---

## Clases Detectadas por el Modelo (16 Clases)

| Perfil | Elementos Válidos (Verde) | Infracciones / Faltantes (Rojo) |
| :--- | :--- | :--- |
| **Médico** | `DelantalMedico`, `barbijoMedico`, `cofiaMedico`, `guantesMedicos` | `noBarbijo`, `noCofia`, `noCofianoBarbijo`, `noDelantalMedico`, `noGuantes` |
| **Industrial** | `barbijoIndustrial`, `chalecoSeguridad`, `guantesIndustriales`, `proteccionAuditiva` | `noBarbijo`, `noBarbijoNoProteccionAudictiva`, `noChalecoSeguridad`, `noGuantes`, `noProteccionAuditiva` |

---

## Estructura del Proyecto

```text
RobotSeguridad/
├── entorno/                        # Simulador MuJoCo + Modelos oficiales Unitree (G1 y Go2)
├── mi_desarrollo/
│   ├── tpRobot.py                  # Programa principal del Agente (consola + conexión al robot)
│   ├── detector_seguridad.py       # Módulo de visión YOLOv8, Modo Escáner y registro de evidencias
│   ├── ejecutor.py                 # Traductor de intenciones a velocidades y tiempos del SDK
│   ├── robot.py                    # Conector con el simulador
│   ├── entrenar_yolo.py            # Script de entrenamiento con YOLOv8s (GPU/CPU automático)
│   ├── train_v2/                   # Modelo principal (yolov8s) -> weights/best.pt y métricas
│   ├── train_nano_backup/          # Respaldo del primer modelo entrenado (yolov8n)
│   ├── evidencias/                 # Capturas .jpg generadas automáticamente en cada inspección
│   └── registro_accesos.csv        # Historial de auditoría (fecha, modo, veredicto, detecciones)
├── INICIAR_SIMULADOR.bat           # Levanta la ventana 3D del robot (Windows)
└── EJECUTAR_MI_CODIGO.bat          # Ejecuta mi_desarrollo/tpRobot.py (Windows)
```

---

## Requisitos e Instalación

En la computadora o notebook donde se vaya a correr el proyecto, instalar las dependencias de Python:

```bash
pip install ultralytics opencv-python mujoco
```

*(No hace falta volver a entrenar el modelo en cada computadora: el archivo ya entrenado está incluido en `mi_desarrollo/train_v2/weights/best.pt`).*

---

## Cómo Ejecutar el Proyecto (Paso a Paso)

### 1. Abrir el Simulador 3D
1. Ejecutar **`INICIAR_SIMULADOR.bat`**.
2. Elegir la opción **`1` (Unitree G1 - humanoide)** y presionar `Enter`.
3. Esperar a que se abra la ventana 3D con el robot de pie y dejarla abierta.

### 2. Ejecutar el Agente (`tpRobot.py`)
1. Ejecutar **`EJECUTAR_MI_CODIGO.bat`** (o por terminal: `python mi_desarrollo/tpRobot.py`).
2. Una vez conectado al robot, escribir cualquiera de los siguientes comandos en la consola (`>`):

#### Comandos de Visión y Seguridad (YOLO):
* **`camara en vivo medico`** $\rightarrow$ Abre el escáner en vivo configurado para equipo médico.
* **`camara en vivo industrial`** $\rightarrow$ Abre el escáner en vivo configurado para equipo industrial.
* **`camara en vivo`** $\rightarrow$ Abre el escáner en vivo evaluando todas las clases.
* **`inspeccionar medico`** / **`inspeccionar industrial`** $\rightarrow$ Realiza un único escaneo de 2.8 segundos, cierra la cámara y ejecuta la reacción del robot.

#### Controles de Teclado dentro de la Ventana de la Cámara:
* **`[ESPACIO]`** o **`[E]`**: Inicia el **barrido del láser de escaneo** (2.8s), emite el veredicto (`ACCESO PERMITIDO` / `ACCESO DENEGADO`) y hace reaccionar al robot en el simulador.
* **`[A]`**: Activa o desactiva el **Auto-Scanner** (escanea automáticamente cada 10 segundos con cuenta regresiva).
* **`[M]`**: Cambia en vivo a **Modo Médico**.
* **`[I]`**: Cambia en vivo a **Modo Industrial**.
* **`[T]`**: Cambia en vivo a **Modo Todos**.
* **`[Q]`**: Cierra la ventana de la cámara y vuelve a la consola de comandos.

#### Comandos de Movimiento por Consola:
* `avanza 2 metros` / `retrocede 1 metro despacio`
* `gira 90 grados a la derecha` / `da media vuelta`
* `saluda` / `frena` / `estado`

---

## Entrenamiento del Modelo (`entrenar_yolo.py`)

El modelo fue entrenado a partir de un dataset propio etiquetado en **Roboflow** (663 imágenes tomadas con los elementos reales de prueba):
* **División del Dataset:** **90% Entrenamiento** (`597` imágenes en `dataSet/train/`) y **10% Validación** (`66` imágenes en `dataSet/valid/`), evitando *data leakage* y *overfitting*.
* **Modelo Base:** `yolov8s.pt` (*Transfer Learning*, 50 épocas, `imgsz=640`, entrenado en GPU NVIDIA GeForce RTX 3090).
* **Métricas obtenidas:** Guardadas junto con las curvas de pérdida, curva F1, curva Precisión-Recall y Matriz de Confusión dentro de `mi_desarrollo/train_v2/`.
* **Plan B para notebooks de bajos recursos:** En `mi_desarrollo/train_nano_backup/weights/best.pt` se conserva también la versión ultraliviana entrenada con `yolov8n.pt`.

