import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def calibrar_homografia(video_path, ancho_lineas, ancho_carril, titulo_ventana="CALIBRAR EJE X E Y"):
    print("Ruta recibida:", video_path)
    print("Existe el archivo?:", os.path.exists(video_path))

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("No se pudo leer el primer frame para calibrar.")

    puntos = []

    def aplicar_overlay(img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 100), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 1, img, 0.25, 0)
        cv2.putText(
            img, "Click izq= punto | ENTER = terminar",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            "Ancho carril: A izq lejos, B dcha lejos, C izq cerca, D dcha cerca",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            "Distancia lineas discontinuas: E final linea, F principio siguiente",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return img

    punto_inicio = None

    def click_event(event, x, y, flags, param):
        nonlocal puntos, frame, punto_inicio

        # CLIC IZQUIERDO → marcar punto
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 6:
            puntos.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Si es el primer punto de la pareja (A o C)
            if len(puntos) in (1, 3, 5):
                punto_inicio = (x, y)  # guardamos el origen de la línea dinámica

            # Si es el segundo punto de la pareja (B o D)
            if len(puntos) in (2, 4):
                # Dibujar línea definitiva entre los dos puntos
                cv2.line(frame, puntos[-2], puntos[-1], (255, 0, 0), 2)
                punto_inicio = None

            if len(puntos) == 6:
                cv2.line(frame, puntos[-2], puntos[-1], (0, 255, 255), 2)
                punto_inicio = None

            ventana = aplicar_overlay(frame.copy())
            cv2.imshow(titulo_ventana, ventana)
            return

        # MOVER RATÓN → dibujar línea desde el primer punto hasta el cursor
        if event == cv2.EVENT_MOUSEMOVE and punto_inicio is not None:
            temp = frame.copy()
            cv2.line(temp, punto_inicio, (x, y), (255, 0, 0), 2)
            ventana = aplicar_overlay(temp)
            cv2.imshow(titulo_ventana, ventana)
            return

        ventana = aplicar_overlay(frame.copy())
        cv2.imshow(titulo_ventana, ventana)

    cv2.namedWindow(titulo_ventana, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(titulo_ventana, aplicar_overlay(frame.copy()))
    cv2.setMouseCallback(titulo_ventana, click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # ENTER
            break

    cv2.destroyAllWindows()

    if len(puntos) != 6:
        raise RuntimeError("Debes marcar exactamente 6 puntos.")

    # HOMOGRAFÍA PROVISIONAL UNIDADES ABSTRACTAS
    pts_img = np.array(puntos[:4], dtype=np.float32)
    pts_real = np.array(
        [
            [0, 1.0],  # A izquierda lejos
            [1.0, 1.0],  # B derecha lejos
            [0, 0],  # C izquierda cerca
            [1.0, 0],  # D derecha cerca
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(pts_img, pts_real)

    # ---PUNTOS A Y B ANCHO CARRIL-----
    (Ax, Ay), (Bx, By) = puntos[0], puntos[1]

    pA = np.array([Ax, Ay, 1.0])
    pB = np.array([Bx, By, 1.0])
    # TRANSFORMAR PUNTOS OCN HOMOGRAFÍA PROVISIONAL
    PA = H @ pA;
    PA /= PA[2]
    PB = H @ pB;
    PB /= PB[2]
    # DISTANCIA EN EL PLANO FALSO
    dX_calc = abs(PB[0] - PA[0])
    # FACTOR REAL A ESCALA REAL
    scale_x = ancho_carril / dX_calc

    # ---PUNTOS E Y F ANCHO ESPACIO ENTRE LÍNEAS DISCONTINUAS--
    (Ex, Ey), (Fx, Fy) = puntos[4], puntos[5]

    pE = np.array([Ex, Ey, 1.0])
    pF = np.array([Fx, Fy, 1.0])

    PE = H @ pE;
    PE /= PE[2]
    PF = H @ pF;
    PF /= PF[2]

    dY_calc = abs(PF[1] - PE[1])
    scale_y = ancho_lineas / dY_calc  # metros por unidad en Y

    return H, scale_x, scale_y


def dibujar_poligonos(video_path, titulo_ventana="CALIBRAR ZONA DE AFORO"):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame para dibujar zonas.")

    puntos = []
    puntos_inicio = None

    def aplicar_overlay(img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 1, img, 0.2, 0)
        cv2.putText(
            img,
            " Click Izq=punto | ENTER=terminar ",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
        return img

    def click_event(event, x, y, flags, param):
        nonlocal puntos, frame, puntos_inicio
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if len(puntos) in (1, 3):
                puntos_inicio = (x, y)
            if len(puntos) in (2, 4):
                cv2.line(frame, puntos[-2], puntos[-1], (0, 255, 0), 3)
                puntos_inicio = None

            ventana = aplicar_overlay(frame.copy())
            cv2.imshow(titulo_ventana, ventana)
            return

        if event == cv2.EVENT_MOUSEMOVE and puntos_inicio is not None:
            temp = frame.copy()
            cv2.line(temp, puntos_inicio, (x, y), (0, 255, 0), 3)
            ventana = aplicar_overlay(temp)
            cv2.imshow(titulo_ventana, ventana)
            return

    cv2.namedWindow(titulo_ventana, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(titulo_ventana, aplicar_overlay(frame.copy()))
    cv2.setMouseCallback(titulo_ventana, click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):
            break

    cv2.destroyAllWindows()
    if len(puntos) != 4: raise RuntimeError("Debes marcar exactamente 4 puntos (2 líneas).")
    linea1 = (puntos[0], puntos[1])
    linea2 = (puntos[2], puntos[3])

    return linea1, linea2, puntos


def punto_en_poligono(x, y, poligono):
    pts = np.array(poligono, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0


def punto_entre_lineas(cx, cy, linea1, linea2):
    # Extraemos las coordenadas Y de cada línea
    y1 = (linea1[0][1] + linea1[1][1]) / 2
    y2 = (linea2[0][1] + linea2[1][1]) / 2

    # Ordenamos por si están invertidas
    ymin = min(y1, y2)
    ymax = max(y1, y2)

    # ¿Está el punto entre ambas?
    return ymin <= cy <= ymax


def procesar_video_trafico(video_path, ancho_lineas, ancho_carril, output_json, output_video=None, max_segundos=20):
    # === 1.CALIBRACIÓN ===
    linea1, linea2, poligono_carretera = dibujar_poligonos(video_path)
    zona_trafico = (linea1, linea2)
    H, scale_x, scale_y = calibrar_homografia(video_path, ancho_lineas, ancho_carril)

    # === 2. MODELO Y VÍDEO ===
    model = YOLO("yolov8s.pt")
    model.fuse()

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    ####PARA TESTEAR, BORRAR LUEGO#####
    # NUEVO: Límite de frames para max_segundos
    max_frames = int(fps * max_segundos)
    ####################################
    video_writer = None
    if output_video:
        video_writer = cv2.VideoWriter(
            output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    # === ESTADO ===
    history = defaultdict(lambda: deque(maxlen=10))
    vehiculos_detectados = {}
    contador_global = 0
    events = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        print(f"PROGRESS {frame_idx}/{max_frames}", flush=True)

        results = model.track(
            frame,
            persist=True,
            conf=0.35,
            imgsz=416,
            verbose=False,
        )
        annotated = results[0].plot() if video_writer else frame.copy()

        # DIBUJAR POLIGONOS DE ZONA
        for idx, (p1, p2) in enumerate(zona_trafico, start=1):
            cv2.line(annotated, p1, p2, (0, 255, 0), 4)
            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)
            cv2.putText(
                annotated,
                f"Linea {idx}",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = y2
                if not punto_en_poligono(cx, cy, poligono_carretera): continue
                nombre = model.names[int(cls)]

                # Guardar historial
                history[track_id].append((frame_idx, cx, cy))

                # === ENTRADA ENTRE LÍNEAS ===
                linea1, linea2 = zona_trafico
                esta_en_zona = punto_entre_lineas(cx, cy, linea1, linea2)
                if esta_en_zona and (
                        track_id not in vehiculos_detectados or not vehiculos_detectados[track_id].get("procesado",
                                                                                                       False)):
                    contador_global += 1
                    vehiculos_detectados[track_id] = {
                        "id": contador_global,
                        "frame": frame_idx,
                        "nombre": nombre,
                    }

                if track_id in vehiculos_detectados and len(history[track_id]) >= 3:
                    f1, x1, y1 = history[track_id][0]
                    f2, x2, y2 = history[track_id][-1]

                    dt = (f2 - f1) / fps
                    if dt > 0:

                        # Convertir puntos de imagen → plano real usando homografía
                        p1 = np.array([x1, y1, 1.0])
                        p2 = np.array([x2, y2, 1.0])

                        P1 = H @ p1
                        P2 = H @ p2

                        P1 /= P1[2]
                        P2 /= P2[2]

                        dx = P2[0] - P1[0]
                        dy = P2[1] - P1[1]

                        dx_m = dx * scale_x
                        dy_m = dy * scale_y

                        # Filtro anti-saltos imposibles
                        if abs(dx_m) > 3 or abs(dy_m) > 15:
                            continue

                        d_metros = np.hypot(dx_m, dy_m)
                        v_kmh = (d_metros / dt) * 3.6

                        if v_kmh >= 10 and v_kmh <= 150:
                            peso = "pesado" if nombre in ("bus", "truck") else "ligero"
                            events.append(
                                {
                                    "track_id": int(track_id),
                                    "clase_nombre": vehiculos_detectados[track_id]["nombre"],
                                    "peso": peso,
                                    "velocidad_kmh": round(v_kmh, 1),
                                    "timestamp_s": round(frame_idx / fps, 1),
                                }
                            )
                            vehiculos_detectados[track_id]["procesado"] = True

                if video_writer:
                    color = (
                        (0, 0, 255)
                        if track_id in vehiculos_detectados
                        else ((0, 255, 0) if esta_en_zona else (255, 255, 255))
                    )
                    cv2.circle(annotated, (int(cx), int(cy)), 8, color, 2)

        if video_writer:
            video_writer.write(annotated)

        frame_idx += 1

    # JSON FINAL
    cap.release()
    if video_writer:
        video_writer.release()
    # POST-PROCESADO: 1 por track_id con velocidad media
    vel_por_track = defaultdict(list)
    for event in events:
        if event["clase_nombre"] == "person":
            continue
        vel_por_track[event["track_id"]].append(event["velocidad_kmh"])

    events_final = {}
    for track_id, velocidades in vel_por_track.items():
        v_media = round(float(np.mean(velocidades)), 1)
        base = next(e for e in events if e["track_id"] == track_id)
        base["velocidad_kmh"] = v_media
        events_final[track_id] = base

    # Convertir a lista ordenada
    events_filtrados = list(events_final.values())
    for i, event in enumerate(events_filtrados, 1):
        event["id"] = i  # Reasignar ID secuenciales
    total_ligero = sum(1 for e in events_filtrados if e["peso"] == "ligero")
    total_pesado = sum(1 for e in events_filtrados if e["peso"] == "pesado")
    vel_media = (
        np.mean([e["velocidad_kmh"] for e in events_filtrados])
        if events_filtrados
        else 0
    )
    vel_max = (
        max([e["velocidad_kmh"] for e in events_filtrados]) if events_filtrados else 0
    )

    resultado = {
        "aforo_total": len(events_filtrados),
        "aforo_ligero": total_ligero,
        "aforo_pesado": total_pesado,
        "velocidad_media_kmh": round(vel_media, 1),
        "velocidad_max_kmh": round(vel_max, 1),
        "events": events_filtrados,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\n✅ TERMINADO! {len(events_filtrados)} vehículos")
    return resultado

if __name__ == "__main__":
    path_video = sys.argv[1]
    lineas_discontinuas = float(sys.argv[2])
    carril = float(sys.argv[3])

    downloads = Path.home() / "Downloads"
    carpeta = downloads / "resultados"
    carpeta.mkdir(exist_ok=True)
    nombre_video = Path(path_video).stem
    json_salida = carpeta / f"{nombre_video}_aforo.json"
    video_salida = carpeta / f"{nombre_video}_anotado.mp4"

    resultado_final = procesar_video_trafico(
        video_path=path_video,
        ancho_lineas=lineas_discontinuas,
        ancho_carril=carril,
        output_json=str(json_salida),
        output_video=str(video_salida),
        max_segundos=20,
    )
    print("\n🎉 TODOS LOS VÍDEOS PROCESADOS!")
