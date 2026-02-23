package com.mycompany.aforotraficoapp;

import java.io.IOException;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.collections.*;
import org.json.*;
import javafx.beans.property.ReadOnlyStringWrapper;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ResultadosViewController {

    @FXML
    private TableView<ObservableList<String>> tablaGlobal;
    @FXML
    private TableColumn<ObservableList<String>, String> colGlobalCampo;
    @FXML
    private TableColumn<ObservableList<String>, String> colGlobalValor;

    @FXML
    private TableView<ObservableList<String>> tablaSalidas;
    @FXML
    private TableColumn<ObservableList<String>, String> colSalida;
    @FXML
    private TableColumn<ObservableList<String>, String> colSalidaTotal;
    @FXML
    private TableColumn<ObservableList<String>, String> colSalidaLigero;
    @FXML
    private TableColumn<ObservableList<String>, String> colSalidaPesado;

    @FXML
    private TableView<ObservableList<String>> tablaVehiculos;
    @FXML
    private TableColumn<ObservableList<String>, String> colVehId;
    @FXML
    private TableColumn<ObservableList<String>, String> colVehClase;
    @FXML
    private TableColumn<ObservableList<String>, String> colVehPeso;
    @FXML
    private TableColumn<ObservableList<String>, String> colVehVelocidad;

    public void initialize() {
        configurarTablas();
    }

    private void configurarTablas() {
        // Columnas ocupan todo el ancho
        tablaGlobal.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_ALL_COLUMNS);
        tablaSalidas.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_ALL_COLUMNS);
        tablaVehiculos.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_ALL_COLUMNS);

        // Solo configurar cellValueFactory (CSS hace el resto)
        colGlobalCampo.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(0)));
        colGlobalValor.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(1)));

        colSalida.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(0)));
        colSalidaTotal.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(1)));
        colSalidaLigero.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(2)));
        colSalidaPesado.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(3)));

        colVehId.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(0)));
        colVehClase.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(1)));
        colVehPeso.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(2)));
        colVehVelocidad.setCellValueFactory(data -> new ReadOnlyStringWrapper(data.getValue().get(3)));
    }

    public void cargarJSONRotonda(String ruta) {
        try {
            String contenido = new String(Files.readAllBytes(Paths.get(ruta)));
            JSONObject json = new JSONObject(contenido);

            // ============================
            // TABLA 1: RESUMEN GLOBAL
            // ============================
            ObservableList<ObservableList<String>> filasGlobal = FXCollections.observableArrayList();
            JSONObject global = json.getJSONObject("resumen_global");

            filasGlobal.add(FXCollections.observableArrayList("Total", global.get("total").toString()));
            filasGlobal.add(FXCollections.observableArrayList("Ligeros", global.get("ligero").toString()));
            filasGlobal.add(FXCollections.observableArrayList("Pesados", global.get("pesado").toString()));

            tablaGlobal.setItems(filasGlobal);

            // ============================
            // TABLA 2: RESUMEN SALIDAS
            // ============================
            ObservableList<ObservableList<String>> filasSalidas = FXCollections.observableArrayList();
            JSONObject salidas = json.getJSONObject("resumen_salidas");

            for (String key : salidas.keySet()) {
                JSONObject s = salidas.getJSONObject(key);
                filasSalidas.add(FXCollections.observableArrayList(
                        key,
                        s.get("total").toString(),
                        s.get("ligero").toString(),
                        s.get("pesado").toString()
                ));
            }

            tablaSalidas.setItems(filasSalidas);

            // ============================
            // TABLA 3: VEHÍCULOS
            // ============================
            ObservableList<ObservableList<String>> filasVehiculos = FXCollections.observableArrayList();
            JSONArray aforo = json.getJSONArray("aforo_total");

            for (int i = 0; i < aforo.length(); i++) {
                JSONObject v = aforo.getJSONObject(i);
                filasVehiculos.add(FXCollections.observableArrayList(
                        v.get("track_id").toString(),
                        v.get("clase_nombre").toString(),
                        v.get("peso").toString(),
                        v.get("velocidad_kmh").toString()
                ));
            }

            tablaVehiculos.setItems(filasVehiculos);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void cargarJSONCarretera(String ruta) throws IOException {
        String contenido = new String(Files.readAllBytes(Paths.get(ruta)));
        JSONObject json = new JSONObject(contenido);
        // TABLA 1: RESUMEN GLOBAL
        ObservableList<ObservableList<String>> filasGlobal = FXCollections.observableArrayList();

        filasGlobal.add(FXCollections.observableArrayList("Total", json.get("aforo_total").toString()));
        filasGlobal.add(FXCollections.observableArrayList("Ligeros", json.get("aforo_ligero").toString()));
        filasGlobal.add(FXCollections.observableArrayList("Pesados", json.get("aforo_pesado").toString()));
        filasGlobal.add(FXCollections.observableArrayList("Velocidad media (km/h)", json.get("velocidad_media_kmh").toString()));
        filasGlobal.add(FXCollections.observableArrayList("Velocidad máxima (km/h)", json.get("velocidad_max_kmh").toString()));

        tablaGlobal.setItems(filasGlobal);

        // TABLA 2: SALIDAS → NO EXISTE EN CARRETERA
        tablaSalidas.setItems(FXCollections.observableArrayList());

        // TABLA 3: VEHÍCULOS
        ObservableList<ObservableList<String>> filasVehiculos = FXCollections.observableArrayList();
        JSONArray eventos = json.getJSONArray("events");

        for (int i = 0; i < eventos.length(); i++) {
            JSONObject v = eventos.getJSONObject(i);
            filasVehiculos.add(FXCollections.observableArrayList(
                    v.get("track_id").toString(),
                    v.get("clase_nombre").toString(),
                    v.get("peso").toString(),
                    v.get("velocidad_kmh").toString()
            ));
        }

        tablaVehiculos.setItems(filasVehiculos);
    }
}
