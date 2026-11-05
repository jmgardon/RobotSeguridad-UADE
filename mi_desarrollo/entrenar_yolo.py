from pathlib import Path
import torch
from ultralytics import YOLO


def buscar_data_yaml(base_dir: Path) -> Path:
    ruta_por_defecto = base_dir / "dataSet" / "data.yaml"
    if ruta_por_defecto.exists():
        return ruta_por_defecto

    encontrados = list(base_dir.glob("*/data.yaml"))
    if encontrados:
        return encontrados[0]

    raise FileNotFoundError(
        f"No se encontró ningún archivo 'data.yaml' dentro de {base_dir}.\n"
        "Asegurate de copiar la carpeta de tu dataset (con train/, valid/ y data.yaml) "
        "adentro de 'mi_desarrollo/'."
    )


def entrenar():
    base_dir = Path(__file__).resolve().parent
    ruta_yaml = buscar_data_yaml(base_dir)

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"[Entrenamiento] Usando dataset: {ruta_yaml}")
    print(f"[Entrenamiento] Dispositivo detectado: {dispositivo}")

    modelo = YOLO("yolov8s.pt")

    resultados = modelo.train(
        data=str(ruta_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=dispositivo,
        project=str(base_dir),
        name="train_v2",
        exist_ok=True,
    )

    ruta_best = base_dir / "train_v2" / "weights" / "best.pt"
    print("\n============================================================")
    print(f"Entrenamiento finalizado. Modelo guardado en:\n  {ruta_best}")
    print("============================================================")
    return resultados


if __name__ == "__main__":
    entrenar()

