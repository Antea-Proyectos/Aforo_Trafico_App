# AforoTraficoApp

Aplicación de escritorio desarrollada en **Java + JavaFX** que permite procesar vídeos de tráfico para obtener:

- **Aforo** (conteo de vehículos)
- **Velocidad estimada**
- **Clasificación por tipo de vía**:
  - **Rotondas**
  - **Carreteras**
  - **Autopistas**

La aplicación integra modelos de visión artificial mediante **Python + YOLOv8**, ejecutando scripts embebidos desde Java para analizar los vídeos y generar resultados en formato JSON.

---

## ¿Qué hace la aplicación?

1. El usuario selecciona un vídeo de tráfico.
2. Elige el tipo de vía (rotonda, carretera o autopista).
3. La aplicación ejecuta un script Python que:
   - Detecta vehículos con YOLOv8.
   - Calcula velocidades aproximadas.
   - Cuenta entradas y salidas (especialmente en rotondas).
   - Genera un archivo JSON con los resultados.
4. La interfaz JavaFX muestra los resultados procesados.

---

## 📂 Estructura del proyecto

```
AforoTraficoApp/
│
├── src/
│   ├── main/java/…         # Código Java y controladores JavaFX
│   └── main/resources/…    # FXML, CSS y recursos
│
├── scripts/                # Scripts Python que procesan los vídeos
│   ├── CodigoFinal.py
│   └── rotonda.py
│
├── python/                 # Entorno Python embebido (NO incluido en el repo)
│   ├── Lib/
│   ├── Scripts/
│   ├── site-packages/
│   └── python.exe
│
├── pom.xml                  # Configuración Maven
├── nbactions.xml            # Configuración NetBeans
└── .gitignore
```

---

## Python embebido (IMPORTANTE)

El proyecto **no incluye la carpeta `python/`** porque contiene:

- `python.exe`
- `site-packages/`
- librerías instaladas
- dependencias pesadas
- el entorno completo necesario para ejecutar YOLO

Esta carpeta pesa demasiado para GitHub y **no debe subirse al repositorio**.

### ¿Cómo obtenerla?

La carpeta completa `python/` estará disponible como **Release** en este repositorio.

El usuario debe:

1. Descargar el archivo `.zip` desde la sección **Releases**.
2. Extraerlo en la raíz del proyecto, quedando así:
AforoTraficoApp/python/

3. Ejecutar la aplicación normalmente.

---
## Requisitos

- **Java 21 o superior**
- **Maven**
- **Windows** (el Python embebido está preparado para Windows)
- **Modelo YOLOv8 (`yolov8s.pt`)**  
  → No se incluye en el repo.  
  → Debe colocarse en la raíz del proyecto.

---

## Ejecución

Desde NetBeans o desde terminal:

```bash
mvn javafx:run
```
---
## Licencia

Este proyecto se distribuye bajo la licencia **GNU**. Consulta [LICENSE](LICENSE).
