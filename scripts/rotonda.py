from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import os


def dibujar_poligonos(video_path, titulo_ventana):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame para dibujar zonas.")

    dibujo = frame.copy()
    poligonos = []
    puntos_actual = []

    def click_event(event, x, y, flags, param):
        nonlocal puntos_actual, dibujo
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos_actual.append((x, y))
            cv2.circle(dibujo, (x, y), 5, (0, 255, 0), -1)
            if len(puntos_actual) > 1:
                cv2.line(dibujo, puntos_actual[-2], puntos_actual[-1], (0, 255, 0), 2)
            cv2.imshow(titulo_ventana, dibujo)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(puntos_actual) >= 3:
                poligonos.append(np.array(puntos_actual, dtype=np.int32))
                cv2.polylines(dibujo, [poligonos[-1]], True, (0, 255, 255), 2)
            puntos_actual = []
            cv2.imshow(titulo_ventana, dibujo)

    cv2.namedWindow(titulo_ventana)
    cv2.putText(
        dibujo,
        "Izq=punto | Der=cerrar poligono | ENTER=terminar",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(titulo_ventana, dibujo)
    cv2.setMouseCallback(titulo_ventana, click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # ENTER
            break

    cv2.destroyAllWindows()
    return poligonos

def dentro_poligono(cx, cy, poligono):
    return cv2.pointPolygonTest(poligono, (cx, cy), False) >= 0

def centroide(poligono):
    M = cv2.moments(poligono)
    if M["m00"] == 0:
        return poligono[0][0], poligono[0][1]
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def calibrar_meter_per_pixel(video_path, metros_reales=5, titulo_ventana="CALIBRAR ANCHO CARRIL"):
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
            cv2.imshow(titulo_ventana, frame)

    cv2.namedWindow(titulo_ventana)
    
    cv2.putText(
        frame,
        "2 clicks sobre lineas que delimitan un carril | ENTER=terminar",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(titulo_ventana, frame)
    cv2.setMouseCallback(titulo_ventana, click_event)

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

def estabilizar_frame(prev_gray, curr_gray, prev_frame):
    # Detectar puntos buenos en el frame anterior
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)

    if prev_pts is None:
        return curr_gray, prev_frame  # no hay puntos, no estabilizamos

    # Calcular flujo óptico
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Filtrar puntos válidos
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Calcular transformación afín
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    if m is None:
        return curr_gray, prev_frame

    # Aplicar transformación al frame actual
    stabilized = cv2.warpAffine(prev_frame, m, (prev_frame.shape[1], prev_frame.shape[0]))

    return curr_gray, stabilized


def procesar_video_rotonda(
    video_path,
    output_json,
    output_video,
):
    print("Dibuja las ZONAS DE ENTRADA")
    zonas_entrada = dibujar_poligonos(video_path, "ZONAS ENTRADA")

    print("Dibuja las ZONAS DE SALIDA")
    zonas_salida = dibujar_poligonos(video_path, "ZONAS SALIDA")
    
    print("Calibrar ANCHO DEL ARCEN") 
    meter_per_pixel = calibrar_meter_per_pixel(video_path, metros_reales=5)

    entradas_por_zona = [set() for _ in zonas_entrada]
    salidas_por_zona = [set() for _ in zonas_salida]

    pesos = {}  # track_id → ligero/pesado

    model = YOLO("yolov8s.pt")
    model.fuse()

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = None
    if output_video:
        video_writer = cv2.VideoWriter(
            output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    frame_idx = 0
    prev_gray = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"PROGRESS {frame_idx}/{max_frames}", flush=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # Estabilizar 
        if prev_gray is None:
            prev_gray = gray 
            stabilized = frame.copy() 
        else:
            prev_gray, stabilized = estabilizar_frame(prev_gray, gray, frame) 
        # Usar el frame estabilizado para TODO
        frame = stabilized
        
        results = model.track(
            frame,
            persist=True,
            conf=0.5,
            imgsz=416,
            verbose=False,
            tracker="botsort.yaml",
        )
        annotated = results[0].plot() if video_writer else frame.copy()

        # Dibujar polígonos
        for idx, poly in enumerate(zonas_entrada, start=1):
            cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
            cx, cy = centroide(poly)
            cv2.putText(annotated, f"Entrada {idx}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        for idx, poly in enumerate(zonas_salida, start=1):
            cv2.polylines(annotated, [poly], True, (0, 0, 255), 2)
            cx, cy = centroide(poly)
            cv2.putText(annotated, f"Salida {idx}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

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
                pesos[track_id] = peso

                # ENTRADAS
                for idx_z, poly in enumerate(zonas_entrada):
                    if dentro_poligono(cx, cy, poly):
                        entradas_por_zona[idx_z].add(track_id)

                # SALIDAS
                for idx_z, poly in enumerate(zonas_salida):
                    if dentro_poligono(cx, cy, poly):
                        salidas_por_zona[idx_z].add(track_id)

                cv2.circle(annotated, (int(cx), int(cy)), 6, (255, 255, 255), 2)

        if video_writer:
            video_writer.write(annotated)

        # cv2.imshow("LIVE", annotated)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        frame_idx += 1

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # ==========================
    # RESÚMENES
    # ==========================

    # IDs que han pasado por alguna zona
    ids_global = set().union(*entradas_por_zona, *salidas_por_zona)

    resumen_global = {
        "total": len(ids_global),
        "ligero": sum(1 for tid in ids_global if pesos.get(tid) == "ligero"),
        "pesado": sum(1 for tid in ids_global if pesos.get(tid) == "pesado"),
    }

    entradas_resumen = []
    for idx, zona in enumerate(entradas_por_zona, start=1):
        ids = list(zona)
        entradas_resumen.append({
            "zona": idx,
            "total": len(ids),
            "ligero": sum(1 for tid in ids if pesos.get(tid) == "ligero"),
            "pesado": sum(1 for tid in ids if pesos.get(tid) == "pesado"),
            "track_ids": ids
        })

    salidas_resumen = []
    for idx, zona in enumerate(salidas_por_zona, start=1):
        ids = list(zona)
        salidas_resumen.append({
            "zona": idx,
            "total": len(ids),
            "ligero": sum(1 for tid in ids if pesos.get(tid) == "ligero"),
            "pesado": sum(1 for tid in ids if pesos.get(tid) == "pesado"),
            "track_ids": ids
        })

    resultado = {
        "resumen_global": resumen_global,
        "entradas": entradas_resumen,
        "salidas": salidas_resumen,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    return resultado


if __name__ == "__main__":
    video_path = sys.argv[1]

    downloads = Path.home() / "Downloads"
    carpeta = downloads / "resultados"
    carpeta.mkdir(exist_ok=True)

    nombre = Path(video_path).stem
    output_json = carpeta / f"{nombre}_aforo.json"
    output_video = carpeta / f"{nombre}_anotado.mp4"

    procesar_video_rotonda(
        video_path=video_path,
        output_json=str(output_json),
        output_video=str(output_video),
    )
