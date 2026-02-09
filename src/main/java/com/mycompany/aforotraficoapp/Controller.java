package com.mycompany.aforotraficoapp;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/javafx/FXMLController.java to edit this template
 */
import java.io.BufferedReader;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser;
import java.io.File;
import java.io.InputStreamReader;
import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

/**
 * FXML Controller class
 *
 * @author nereatrillo
 */
public class Controller {

    @FXML
    private Button btnProcesar;
    @FXML
    private Button btnResultados;
    @FXML
    private ComboBox<String> comboTipo;
    @FXML
    private TextField txtRutaVideo;
    private String rutaVideo;
    @FXML
    private TextField txtNumeroSalidas;
    @FXML
    private Label labelSalidas;
    private String numSalidas;
    @FXML
    private ProgressBar progreso;
    @FXML
    private StackPane panelProgreso;
    @FXML
    private Label lblPorcentaje;
    private String rutaJsonGenerado;

    /**
     * Initializes the controller class.
     */
    public void initialize() {
        comboTipo.setPromptText("Selecciona el tipo de carretera");
        comboTipo.getItems().addAll("Rotonda", "Carretera o Autopista Día", "Carretera o Autopista Noche");
        comboTipo.valueProperty().addListener((obs, oldVal, newVal) -> {
            if ("Rotonda".equals(newVal)) {
                txtNumeroSalidas.setVisible(true);
                labelSalidas.setVisible(true);
            } else {
                txtNumeroSalidas.setVisible(false);
                labelSalidas.setVisible(false);
            }
        });
        panelProgreso.setVisible(false);
        btnResultados.setVisible(false);
        progreso.setProgress(0);
        lblPorcentaje.setText("0%");

    }

    @FXML
    private void seleccionarVideo() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Seleccionar video");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("videos MP4", "*.mp4"));

        File file = fc.showOpenDialog(null);

        if (file != null) {
            rutaVideo = file.getAbsolutePath();
            txtRutaVideo.setText(rutaVideo);
        }
        btnProcesar.setVisible(true);
        btnResultados.setVisible(false);
    }

    @FXML
    private void abrirVentanaResultados() {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/ResultadosView.fxml"));
            Parent root = loader.load();
            ResultadosViewController rc = loader.getController();
            rc.cargarJSON(rutaJsonGenerado);
            Stage stage = new Stage();
            stage.setTitle("Resultados del análisis de tráfico del video");
            stage.setScene(new Scene(root));
            stage.sizeToScene();
            stage.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @FXML
    private void procesarVideo() {
        if (rutaVideo == null || rutaVideo.isEmpty()) {
            System.out.println("No se ha seleccionado ningún video");
            return;
        }

        String tipo = comboTipo.getValue();
        if (tipo == null) {
            System.out.println("Selecciona un tipo de carretera");
            return;
        }
        btnProcesar.setVisible(false);
        panelProgreso.setVisible(true);
        progreso.setProgress(0);
        lblPorcentaje.setText("0%");

        String nombreVideo = new File(rutaVideo).getName();
        String nombreSinExtension = nombreVideo.substring(0, nombreVideo.lastIndexOf("."));

        String rutaJson = System.getProperty("user.home")
                + "/Downloads/resultados/" + nombreSinExtension + "_salidas.json";

        rutaJsonGenerado = rutaJson;

        new Thread(() -> {
            try {
                // Ruta base donde está el .jar
                String base = new File("").getAbsolutePath();

                // Python embebido
                String pythonPath = base + "/python/python.exe";

                // Script a ejecutar
                String scriptPath;

                String comando;

                if (tipo.contains("Rotonda")) {
                    numSalidas = txtNumeroSalidas.getText();
                    if (numSalidas == null || numSalidas.isEmpty()) {
                        System.out.println("Introduce el número de salidas");
                        return;
                    }

                    scriptPath = base + "/scripts/rotonda.py";
                    comando = "\"" + pythonPath + "\" \"" + scriptPath + "\" \"" + rutaVideo + "\" " + numSalidas;

                } else { // Carretera
                    scriptPath = base + "/scripts/CodigoFinal.py";
                    comando = "\"" + pythonPath + "\" \"" + scriptPath + "\" \"" + rutaVideo + "\"";
                }

                System.out.println("Ejecutando: " + comando);

                Process proceso = Runtime.getRuntime().exec(comando);

                // Capturar errores del script
                BufferedReader stdOut = new BufferedReader(new InputStreamReader(proceso.getInputStream()));
                BufferedReader stdErr = new BufferedReader(new InputStreamReader(proceso.getErrorStream()));

                //------------------------
                //LECTURA PROGESO EN TIEMPO REAL
                //------------------------
                new Thread(() -> {
                    try {
                        String linea;
                        while ((linea = stdOut.readLine()) != null) {
                            System.out.println("PYTHON: " + linea);
                            if (linea.startsWith("PROGRESS")) {
                                String[] partes = linea.split(" ");
                                String[] nums = partes[1].split("/");
                                double actual = Double.parseDouble(nums[0]);
                                double total = Double.parseDouble(nums[1]);
                                double valor = actual / total;
                                int porcentaje = (int) (valor * 100);
                                Platform.runLater(() -> {
                                    progreso.setProgress(valor);
                                    lblPorcentaje.setText(porcentaje + "%");
                                });
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();

                //---------------
                // LECTURA DE ERRORES
                //---------------
                new Thread(() -> {
                    try {
                        String errLine;
                        while ((errLine = stdErr.readLine()) != null) {
                            System.err.println("PYTHON ERROR: " + errLine);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }).start();

                proceso.waitFor();
                System.out.println("Script finalizado");

            } catch (Exception e) {
                e.printStackTrace();
            }
            // Volver al hilo de JavaFX 
            Platform.runLater(() -> {
                progreso.setProgress(1.0);
                lblPorcentaje.setText("100%");
                btnResultados.setVisible(true);
            });
        }
        ).start();
    }
}
