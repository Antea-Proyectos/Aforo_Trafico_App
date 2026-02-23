from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
import sys


def dibujar_poligonos(video_path, titulo_ventana):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame para dibujar zonas.")

    dibujo = frame.copy()
    poligonos = []
    puntos_actual = []

    def aplicar_overlay(img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 1, img, 0.25, 0)
        cv2.putText(
            img, "Click izq=punto | Click dcha=cerrar poligono | ENTER=terminar",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return img

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
            cv2.imshow(titulo_ventana, aplicar_overlay(dibujo))

    cv2.namedWindow(titulo_ventana)
    cv2.imshow(titulo_ventana, aplicar_overlay(dibujo))
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

def procesar_video_rotonda(
        video_path,
        output_json,
        output_video,
):
    zonas_entrada = dibujar_poligonos(video_path, "ZONAS ENTRADA")
    zonas_salida = dibujar_poligonos(video_path, "ZONAS SALIDA")

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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

        # Dibujar polígonos
        for idx, poly in enumerate(zonas_entrada, start=1):
            cv2.polylines(annotated, [poly], True, (0, 255, 0), 2)
            cx, cy = centroide(poly)
            cv2.putText(annotated, f"Entrada {idx}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for idx, poly in enumerate(zonas_salida, start=1):
            cv2.polylines(annotated, [poly], True, (0, 0, 255), 2)
            cx, cy = centroide(poly)
            cv2.putText(annotated, f"Salida {idx}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

    # Id que han pasado por alguna zona
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
    path_video = sys.argv[1]

    downloads = Path.home() / "Downloads"
    carpeta = downloads / "resultados"
    carpeta.mkdir(exist_ok=True)

    nombre_video = Path(path_video).stem
    json_salida = carpeta / f"{nombre_video}_aforo.json"
    video_salida = carpeta / f"{nombre_video}_anotado.mp4"

    procesar_video_rotonda(
        video_path=path_video,
        output_json=str(json_salida),
        output_video=str(video_salida),
    )
