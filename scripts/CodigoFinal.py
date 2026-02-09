from ultralytics import YOLO
import cv2
import numpy as np
import json
from collections import defaultdict, deque
import os

def distancia_a_linea(cx, cy, linea):
    """Distancia del centro del coche a la línea ENTERA"""
    x1, y1 = linea[0]
    x2, y2 = linea[1]
    num = abs((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1)
    den = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return num / den


def calibrar_meter_per_pixel(video_path, metros_reales=3.5):
    """Calibración rápida: 2 clics para obtener meter_per_pixel"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
   
    puntos = []
   
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
            puntos.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            if len(puntos) == 2:
                cv2.line(frame, puntos[0], puntos[1], (255, 0, 0), 2)
                cv2.putText(frame, f"{metros_reales}m", ((puntos[0][0]+puntos[1][0])//2, (puntos[0][1]+puntos[1][1])//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow("CALIBRAR ANCHO CARRIL", frame)
            cv2.waitKey(3000)  # ← 3 SEGUNDOS
   
    cv2.namedWindow("CALIBRAR ANCHO CARRIL")

    cv2.setMouseCallback("CALIBRAR ANCHO CARRIL", click_event)
    # GLOBO INFO
    cv2.rectangle(frame, (20, 30), (frame.shape[1]-20, 160), (0,0,0), -1)  # Fondo negro
    cv2.rectangle(frame, (20, 30), (frame.shape[1]-20, 160), (0,255,255), 3)  # Borde amarillo
    cv2.putText(frame, "CALIBRAR ANCHO DEL CARRIL", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(frame, "Haz 2 clics sobre las lineas que delimitan un carril", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"para medir su ancho real. Distancia real= {metros_reales}m", (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Pulsa ENTER al finalizar", (60, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("CALIBRAR ANCHO CARRIL", frame)
   
    while True:
       key = cv2.waitKey(1) & 0xFF
       # ENTER: 13 en Windows, 10 en otros
       if key in (13, 10):
               break
       elif key == 27:  # ESC para cancelar
           break

    cv2.destroyAllWindows()

    dist_pix = np.hypot(puntos[1][0]-puntos[0][0], puntos[1][1]-puntos[0][1])
    meter_per_pixel = metros_reales / dist_pix
    return meter_per_pixel

def calibrar_lineas(video_path):
    """Calibrar 4 clics: L1 inicio, L1 fin, L2 inicio, L2 fin"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    line1_puntos = []
    line2_puntos = []
    lines_dibujadas = False
    
    def click_event(event, x, y, flags, param):
        nonlocal lines_dibujadas
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(line1_puntos) < 2:
                line1_puntos.append((x, y))
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Verde L1
                if len(line1_puntos) == 2:
                    cv2.line(frame, line1_puntos[0], line1_puntos[1], (0, 255, 0), 3)
                    cv2.putText(frame, "L1 OK", (line1_puntos[0][0], line1_puntos[0][1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            elif len(line2_puntos) < 2:
                line2_puntos.append((x, y))
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Rojo L2
                if len(line2_puntos) == 2:
                    cv2.line(frame, line2_puntos[0], line2_puntos[1], (0, 0, 255), 3)
                    cv2.putText(frame, "L2 OK", (line2_puntos[0][0], line2_puntos[0][1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    lines_dibujadas = True
     
        cv2.imshow("CALIBRAR LINEAS", frame)
        cv2.waitKey(3000) 

    cv2.namedWindow("CALIBRAR LINEAS")
    cv2.setMouseCallback("CALIBRAR LINEAS", click_event)
    # GLOBO INFO 
    cv2.rectangle(frame, (20, 30), (frame.shape[1]-20, 180), (0,0,0), -1)  # Fondo negro
    cv2.rectangle(frame, (20, 30), (frame.shape[1]-20, 180), (0,255,255), 3)  # Borde amarillo
    
    cv2.putText(frame, "CALIBRAR ZONA DE AFORO", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(frame, "Dibuja 2 lineas que crucen toda la carretera, ", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "haciendo click en cada extremo para hacer las lineas", (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Se contaran los coches que crucen entre esas lineas", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "Pulsa ENTER al finalizar", (60, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
  
    cv2.imshow("CALIBRAR LINEAS", frame)
    
    while True:
       key = cv2.waitKey(1) & 0xFF
       # ENTER: 13 en Windows, 10 en otros
       if key in (13, 10):
               break
       elif key == 27:  # ESC para cancelar
           break
    cv2.destroyAllWindows()
    
    return line1_puntos, line2_puntos

def crear_zona_entre_lineas(line1_puntos, line2_puntos):
    """Crea polígono ESPACIO entre L1 y L2 (verticales/diagonales)"""
    p1 = line1_puntos[0]  # L1 inicio
    p2 = line1_puntos[1]  # L1 fin
    p3 = line2_puntos[1]  # L2 fin
    p4 = line2_puntos[0]  # L2 inicio
    return [p1, p2, p3, p4]

def punto_en_zona(cx, cy, zona):
    """¿Centro del vehículo está DENTRO del polígono?"""
    return cv2.pointPolygonTest(np.array(zona, dtype=np.int32), 
                               (int(cx), int(cy)), False) >= 0

def procesar_video_trafico(video_path, output_json="trafico.json", output_video=None, max_segundos=20):
    """
    2 LÍNEAS CALIBRADAS por el usuario con clics - ZONA ENTRE LÍNEAS
    """
    # === 1. CALIBRAR LINEAS + METER/PIXEL ===
    line1_puntos, line2_puntos = calibrar_lineas(video_path)
    METER_PER_PIXEL = calibrar_meter_per_pixel(video_path, metros_reales=3.5)
    zona_trafico = crear_zona_entre_lineas(line1_puntos, line2_puntos)  # ← ZONA

    # === 2. MODELO Y VÍDEO ===
    model = YOLO("yolov8s.pt")
    model.fuse()
    
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ####PARA TESTEAR, BORRAR LUEGO#####
    # NUEVO: Límite de frames para max_segundos
    max_frames = int(fps * max_segundos)
    ####################################    
    video_writer = None
    if output_video:
        video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # === ESTADO ===
    history = {}
    vehiculos_l1 = {}
    contador_global = 0
    events = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= max_frames:
            break

        results = model.track(frame, persist=True, conf=0.35, imgsz=416, verbose=False)
        annotated = results[0].plot() if video_writer else frame.copy()
        
        # DIBUJAR LÍNEAS + ZONA
        cv2.line(annotated, line1_puntos[0], line1_puntos[1], (0, 255, 0), 4)  # Verde L1
        cv2.line(annotated, line2_puntos[0], line2_puntos[1], (0, 0, 255), 4)  # Rojo L2
        cv2.putText(annotated, "L1", line1_puntos[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(annotated, "L2", line2_puntos[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.polylines(annotated, [np.array(zona_trafico, np.int32)], True, (255,255,0), 3)  # ← AMARILLO

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                nombre = model.names[int(cls)]

                if track_id not in history:
                    history[track_id] = deque(maxlen=10)
                history[track_id].append((frame_idx, cx, cy))

                # === ZONA ENTRE LÍNEAS (SOLO ESTO) ===
                if punto_en_zona(cx, cy, zona_trafico) and track_id not in vehiculos_l1:
                    contador_global += 1
                    vehiculos_l1[track_id] = {"id": contador_global, "frame": frame_idx, "nombre": nombre}

                # Velocidad (cualquier vehículo en zona con historia)
                if track_id in vehiculos_l1 and len(history[track_id]) >= 2:
                    veh_l1 = vehiculos_l1[track_id]
                    f1, x1, y1 = history[track_id][-2]
                    f2, x2, y2 = history[track_id][-1]
                    d_pix = np.hypot(x2 - x1, y2 - y1)
                    
                    if d_pix >= 5:
                        dt = (f2 - f1) / fps
                        if dt > 0:
                            v_kmh = (d_pix * METER_PER_PIXEL / dt) * 3.6
                            if 3 < v_kmh < 300:
                                peso = "pesado" if nombre in ("bus", "truck") else "ligero"
                                events.append({
                                    "id": veh_l1["id"],
                                    "track_id": int(track_id),
                                    "clase_nombre": nombre,
                                    "peso": peso,
                                    "velocidad_kmh": round(v_kmh, 1),
                                    "timestamp_s": round(frame_idx / fps, 1)
                                })
                                del vehiculos_l1[track_id]

                if video_writer:
                    color = (0, 0, 255) if track_id in vehiculos_l1 else (0, 255, 0) if punto_en_zona(cx, cy, zona_trafico) else (255,255,255)
                    cv2.circle(annotated, (int(cx), int(cy)), 8, color, 2)

        if video_writer:
            video_writer.write(annotated)

        frame_idx += 1

    # JSON FINAL
    cap.release()
    if video_writer:
        video_writer.release()
    # POST-PROCESADO: 1 por track_id con velocidad máxima
    events_final = {}
    for event in events:
        track_id = event["track_id"]
        
        if event["clase_nombre"] == "person":
            continue
    
        if track_id not in events_final or event["velocidad_kmh"] > events_final[track_id]["velocidad_kmh"]:
            events_final[track_id] = event

    # Convertir a lista ordenada
    events_filtrados = list(events_final.values())
    for i, event in enumerate(events_filtrados, 1):
        event["id"] = i  # Reasignar IDs secuenciales
    total_ligero = sum(1 for e in events_filtrados if e["peso"] == "ligero")
    total_pesado = sum(1 for e in events_filtrados if e["peso"] == "pesado")
    vel_media = np.mean([e["velocidad_kmh"] for e in events_filtrados]) if events_filtrados else 0
    vel_max = max([e["velocidad_kmh"] for e in events_filtrados]) if events_filtrados else 0

    resultado = {
        "aforo_total": len(events_filtrados),
        "aforo_ligero": total_ligero,
        "aforo_pesado": total_pesado,
        "velocidad_media_kmh": round(vel_media, 1),
        "velocidad_max_kmh": round(vel_max, 1),
        "events": events_filtrados
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\n✅ TERMINADO! {len(events)} vehículos")
    return resultado

# === USO ===
# === USO MÚLTIPLES VÍDEOS ===
if __name__ == "__main__":
    # Lista de videos
    videos = ["videos/trafico.mp4", "videos/GOPR0050.mp4"]
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n🚗 PROCESANDO VIDEO {i}/{len(videos)}: {video_path}")
        
        resultado = procesar_video_trafico(
            video_path=video_path,
            output_json=f"resultados/aforo_velocidades_{i}.json",
            output_video=f"resultados/salida_anotada_{i}.mp4",
            max_segundos=20
        )
    print("\n🎉 TODOS LOS VÍDEOS PROCESADOS!")

