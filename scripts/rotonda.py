from ultralytics import YOLO
import cv2
import numpy as np
import json
from collections import deque
import os
import sys
from pathlib import Path


# ==========================
# UTILIDADES GEOMÉTRICAS
# ==========================
def distancia_a_linea(cx, cy, linea):
    (x1, y1), (x2, y2) = linea
    num = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
    den = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return num / den


def cruza_segmento(p_prev, p_now, linea, tolerancia=12):
    """
    Detecta si el coche cruza la línea de salida aunque:
    - el cruce ocurra fuera del segmento
    - el cruce ocurra entre frames
    - el cruce ocurra por el extremo
    - la línea sea corta o inclinada
    """

    (x1, y1), (x2, y2) = linea
    x3, y3 = p_prev
    x4, y4 = p_now

    # Vector de la línea
    dxL = x2 - x1
    dyL = y2 - y1

    # Vectores desde la línea a los puntos prev y now
    dx1 = x3 - x1
    dy1 = y3 - y1
    dx2 = x4 - x1
    dy2 = y4 - y1

    # Producto cruzado (lado)
    cross1 = dxL * dy1 - dyL * dx1
    cross2 = dxL * dy2 - dyL * dx2

    # 1) Cambio de lado → cruce real
    if (cross1 > 0 and cross2 < 0) or (cross1 < 0 and cross2 > 0):
        return True

    # 2) Si pasa muy cerca de la línea → también cuenta
    dist_prev = distancia_a_linea(x3, y3, linea)
    dist_now = distancia_a_linea(x4, y4, linea)

    if dist_prev < tolerancia or dist_now < tolerancia:
        return True

    # 3) Si se aleja de la rotonda en dirección a la salida
    if dist_now > dist_prev and dist_prev < tolerancia * 2:
        return True

    return False


# ==========================
# CALIBRACIONES
# ==========================
def calibrar_meter_per_pixel(video_path, metros_reales=5):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(
            "No se pudo leer el primer frame del vídeo para calibrar metros/píxel."
        )

    puntos = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
            puntos.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            if len(puntos) == 2:
                cv2.line(frame, puntos[0], puntos[1], (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{metros_reales}m",
                    (
                        (puntos[0][0] + puntos[1][0]) // 2,
                        (puntos[0][1] + puntos[1][1]) // 2,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            cv2.imshow("CALIBRAR ANCHO CARRIL", frame)

    cv2.namedWindow("CALIBRAR ANCHO CARRIL")
    # ==========================
    # INFO ARRIBA
    # ==========================
    cv2.rectangle(frame, (20, 30), (frame.shape[1] - 20, 160), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 30), (frame.shape[1] - 20, 160), (0, 255, 255), 3)
    cv2.putText(
        frame,
        "CALIBRAR ANCHO DEL CARRIL",
        (60, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Haz 2 clics sobre las lineas que delimitan un carril",
        (60, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"para medir su ancho real. Distancia real= {metros_reales}m",
        (60, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Pulsa ENTER al finalizar",
        (60, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.setMouseCallback("CALIBRAR ANCHO CARRIL", click_event)
    cv2.imshow("CALIBRAR ANCHO CARRIL", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # ENTER: 13 en Windows, 10 en otros
        if key in (13, 10):
            break
        elif key == 27:  # ESC para cancelar
            break

    cv2.destroyAllWindows()

    dist_pix = np.hypot(puntos[1][0] - puntos[0][0], puntos[1][1] - puntos[0][1])
    meter_per_pixel = metros_reales / dist_pix
    return meter_per_pixel


# ---------------------------
# CALIBRAR SALIDAS
# ---------------------------
def calibrar_salidas(video_path, n_salidas):
    """
    Calibrar salidas de la rotonda:
    - Pregunta cuántas salidas hay.
    - Para cada salida: 2 clics sobre la línea de salida.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(
            "No se pudo leer el primer frame del vídeo para calibrar salidas."
        )

    # while True:
    #     try:
    #         n_salidas = int(input("¿Cuántas salidas quieres marcar en la rotonda? "))
    #         if n_salidas <= 0:
    #             continue
    #         break
    #     except ValueError:
    #         print("Introduce un número entero válido.")

    salidas = []
    puntos_actual = []
    salida_idx = 0

    def click_event(event, x, y, flags, param):
        nonlocal salida_idx, puntos_actual
        if event == cv2.EVENT_LBUTTONDOWN and salida_idx < n_salidas:
            puntos_actual.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            if len(puntos_actual) == 2:
                cv2.line(frame, puntos_actual[0], puntos_actual[1], (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"SALIDA {salida_idx+1}",
                    (puntos_actual[0][0], puntos_actual[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                salidas.append((puntos_actual[0], puntos_actual[1]))
                puntos_actual = []
                salida_idx += 1
            cv2.imshow("CALIBRAR SALIDAS", frame)

    cv2.namedWindow("CALIBRAR SALIDAS")
    # ==========================
    # INFO ABAJO A LA DERECHA
    # ==========================
    h, w = frame.shape[:2]
    x1, y1 = w - 580, h - 280
    x2, y2 = w - 20, h - 80
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(
        frame,
        "CALIBRAR SALIDAS ROTONDA",
        (x1 + 10, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Haz 2 clics por cada salida sobre la linea de salida",
        (x1 + 10, y1 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Primero punto y luego punto opuesto",
        (x1 + 10, y1 + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Total salidas a marcar: {n_salidas}",
        (x1 + 10, y1 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Despues de cada salida, pulsa ENTER para finalizar",
        (x1 + 10, y1 + 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.setMouseCallback("CALIBRAR SALIDAS", click_event)
    cv2.imshow("CALIBRAR SALIDAS", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # ENTER: 13 en Windows, 10 en Linux/macOS
        if key in (13, 10):
            break
        # ESC para cancelar
        if key == 27:
            salidas = []
            break

    cv2.destroyAllWindows()

    return salidas


# ==========================
# PROCESADO ROTONDA
# ==========================
def procesar_video_rotonda(
    video_path,
    output_json,
    output_video,
    n_salidas,
    max_segundos=60,
    metros_reales_carril=5,
):
    """
    Procesa un vídeo de rotonda usando POLÍGONO:
    - Calibra polígono de rotonda.
    - Calibra salidas.
    - Calibra metros/pixel.
    - Detecta vehículos dentro y los que salen por cada salida.
    - Calcula velocidades dentro y al salir.
    """
    salidas = calibrar_salidas(video_path, n_salidas)
    METER_PER_PIXEL = calibrar_meter_per_pixel(
        video_path, metros_reales=metros_reales_carril
    )

    model = YOLO("yolov8s.pt")
    model.fuse()

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_segundos)

    video_writer = None
    if output_video:
        video_writer = cv2.VideoWriter(
            output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    history = {}
    eventos_salidas = []
    aforo = []
    salidas_detectadas_ = set()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        print(f"PROGRESS {frame_idx}/{max_frames}", flush=True)

        results = model.track(
            frame,
            persist=True,
            conf=0.5,
            imgsz=416,
            verbose=False,
            tracker="botsort.yaml",
        )
        annotated = results[0].plot() if video_writer else frame.copy()

        # Dibujar salidas
        for i, linea in enumerate(salidas, 1):
            cv2.line(annotated, linea[0], linea[1], (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"S{i}",
                (linea[0][0], linea[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                nombre = model.names[int(cls)]

                if nombre == "person":
                    continue

                peso = "pesado" if nombre in ("bus", "truck") else "ligero"
                # -----HISTORIAL--------
                if track_id not in history:
                    history[track_id] = deque(maxlen=10)
                history[track_id].append((frame_idx, cx, cy))
                # ------VELOCIDAD--------
                if len(history[track_id]) >= 2:
                    f1, x1h, y1h = history[track_id][-2]
                    f2, x2h, y2h = history[track_id][-1]

                    d_pix = np.hypot(x2h - x1h, y2h - y1h)
                    dt = (f2 - f1) / fps if fps > 0 else 0

                    if dt > 0 and d_pix >= 3:
                        v_kmh = (d_pix * METER_PER_PIXEL / dt) * 3.6
                        if 3 < v_kmh < 300:
                            if (
                                track_id not in aforo
                                or v_kmh > aforo[track_id]["velocidad_kmh"]
                            ):
                                aforo.append(
                                    {
                                        "track_id": int(track_id),
                                        "clase_nombre": nombre,
                                        "peso": peso,
                                        "velocidad_kmh": round(v_kmh, 1),
                                        "timestamp_s": round(frame_idx / fps, 1),
                                    }
                                )
                    # ------DETECCIÓN SALIDA--------
                    for idx_s, linea in enumerate(salidas, 1):
                        if (track_id, idx_s) in salidas_detectadas_:
                            continue
                        if cruza_segmento((x1h, y1h), (x2h, y2h), linea):

                            eventos_salidas.append(
                                {
                                    "salida": idx_s,
                                    "track_id": int(track_id),
                                    "peso": peso,
                                }
                            )
                            salidas_detectadas_.add((track_id, idx_s))

                cv2.circle(annotated, (int(cx), int(cy)), 6, (255, 255, 255), 2)

        if video_writer:
            video_writer.write(annotated)

        frame_idx += 1

    cap.release()
    if video_writer:
        video_writer.release()

    # ==========================
    # POST-PROCESADO FINAL
    # ==========================

    # 1. Aforo total: velocidad máxima por track_id
    def filtrar_max_por_track(lista_eventos):
        res = {}
        for e in lista_eventos:
            tid = e["track_id"]
            if tid not in res or e["velocidad_kmh"] > res[tid]["velocidad_kmh"]:
                res[tid] = e
        return res

    aforo_total = filtrar_max_por_track(aforo)

    # 2. Filtrar eventos de salida solo para construir el resumen
    def filtrar_unico_evento(eventos):
        res = {}
        for e in eventos:
            tid = e["track_id"]
            if tid not in res:
                res[tid] = e
        return list(res.values())

    eventos_salidas_filtrado = filtrar_unico_evento(eventos_salidas)

    # 3. Resumen por salida
    resumen_salidas = {}
    for i, _ in enumerate(salidas, 1):
        eventos_s_i = [e for e in eventos_salidas_filtrado if e["salida"] == i]
        resumen_salidas[f"salida_{i}"] = {
            "total": len(eventos_s_i),
            "ligero": sum(1 for e in eventos_s_i if e["peso"] == "ligero"),
            "pesado": sum(1 for e in eventos_s_i if e["peso"] == "pesado"),
        }

    # 4. Resumen global
    aforo_lista = list(aforo_total.values())
    total = len(aforo_lista)
    ligeros = sum(1 for e in aforo_lista if e["peso"] == "ligero")
    pesados = sum(1 for e in aforo_lista if e["peso"] == "pesado")

    resumen_global = {
        "total": total,
        "ligero": ligeros,
        "pesado": pesados,
    }

    # Resultado final
    resultado = {
        "resumen_global": resumen_global,
        "resumen_salidas": resumen_salidas,
        "aforo_total": list(aforo_total.values()),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    return resultado


# ==========================
# USO MÚLTIPLES VÍDEOS
# ==========================
# if __name__ == "__main__":
#     videos = [
#         "videos/GOPR0035.mp4",
#         # "videos/rotonda2.mp4",
#     ]

#     os.makedirs("resultados", exist_ok=True)

#     for i, video_path in enumerate(videos, 1):
#         procesar_video_rotonda(
#             video_path=video_path,
#             output_json=f"resultados/rotonda_aforo_velocidades_{i}.json",
#             output_video=f"resultados/rotonda_salida_anotada_{i}.mp4",
#             max_segundos=60,
#             metros_reales_carril=5,
#         )

#     print("\n🎉 TODAS LAS ROTONDAS PROCESADAS!")
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import os

    # Ruta del vídeo que viene desde Java
    video_path = sys.argv[1]
    n_salidas = int(sys.argv[2])

    # Carpeta de resultados en Downloads
    downloads = Path.home() / "Downloads"
    carpeta = downloads / "resultados"
    carpeta.mkdir(exist_ok=True)

    # Nombre del JSON según el vídeo
    nombre = Path(video_path).stem
    output_json = carpeta / f"{nombre}_salidas.json"
    output_video = carpeta / f"{nombre}_salida_anotada.mp4"

    # Procesar el vídeo
    procesar_video_rotonda(
        video_path=video_path,
        n_salidas=n_salidas,
        output_json=str(output_json),
        output_video=str(output_video),
        max_segundos=60,
        metros_reales_carril=5,
    )
