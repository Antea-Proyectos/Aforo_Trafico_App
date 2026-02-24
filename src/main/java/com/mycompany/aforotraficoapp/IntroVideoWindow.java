/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.aforotraficoapp;

import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.stage.Modality;
import javafx.stage.Stage;
import java.util.prefs.Preferences;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;

/**
 *
 * @author nereatrillo
 */
public class IntroVideoWindow {

    private final Runnable onClose;

    public IntroVideoWindow(Runnable onClose) {
        this.onClose = onClose;
    }

    private String formatTime(javafx.util.Duration time) {
        int minutes = (int) time.toMinutes();
        int seconds = (int) time.toSeconds() % 60;
        return String.format("%02d:%02d", minutes, seconds);
    }

    public void mostrar() {
        Stage stage = new Stage();
        stage.setTitle("Bienvenida");
        stage.setOnCloseRequest(e -> {
            stage.close();
            onClose.run();
        });

        // Ruta del vídeo dentro del proyecto
        String videoPath = getClass().getResource("/videos/Información.mp4").toExternalForm();

        Media media = new Media(videoPath);
        MediaPlayer mediaPlayer = new MediaPlayer(media);
        MediaView mediaView = new MediaView(mediaPlayer);

        //Tamaño del video para que encaje en la ventana
        mediaView.setPreserveRatio(true);
        mediaView.setFitWidth(1280);
        mediaView.setFitHeight(720 - 80);

        stage.widthProperty().addListener((obs, oldVal, newVal) -> {
            mediaView.setFitWidth(newVal.doubleValue());
        });
        stage.heightProperty().addListener((obs, oldVal, newVal) -> {
            mediaView.setFitHeight(newVal.doubleValue() - 80);
        });

        // ----- CONTROLES DEL REPRODUCTOR -----
        // Botón Play/Pausa
        Button playPause = new Button("Pausa");
        playPause.setOnAction(e -> {
            if (mediaPlayer.getStatus() == MediaPlayer.Status.PLAYING) {
                mediaPlayer.pause();
                playPause.setText("Reproducir");
            } else {
                mediaPlayer.play();
                playPause.setText("Pausa");
            }
        });

        // Barra de progreso (ocupa todo el ancho)
        Slider progress = new Slider(0, 100, 0);
        progress.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(progress, javafx.scene.layout.Priority.ALWAYS);

        // Etiqueta de tiempo
        Label tiempoLabel = new Label("00:00 / 00:00");
        tiempoLabel.setStyle("-fx-text-fill: white;");

        // Actualización del tiempo y progreso
        mediaPlayer.currentTimeProperty().addListener((obs, oldTime, newTime) -> {
            double progressValue = newTime.toMillis() / mediaPlayer.getTotalDuration().toMillis() * 100;
            progress.setValue(progressValue);

            String actual = formatTime(newTime);
            String total = formatTime(mediaPlayer.getTotalDuration());
            tiempoLabel.setText(actual + " / " + total);
        });

        // Permitir mover el progreso manualmente
        progress.setOnMousePressed(e -> {
            mediaPlayer.seek(media.getDuration().multiply(progress.getValue() / 100));
        });

        // Contenedor de controles
        HBox controls = new HBox(10, playPause, progress, tiempoLabel);
        controls.setPadding(new Insets(5));
        controls.setStyle("-fx-background-color: rgba(0,0,0,0.6);");

        BorderPane root = new BorderPane();
        root.setCenter(mediaView);
        root.setBottom(controls);
        Scene scene = new Scene(root, 1280, 720);

        stage.setScene(scene);
        stage.setTitle("Reproducción de Video");
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.show();

        mediaPlayer.setOnEndOfMedia(() -> {
            Preferences.userNodeForPackage(Main.class).putBoolean("intro_mostrada", true);
            mediaPlayer.stop();
            stage.close();
            onClose.run();
        });
        mediaPlayer.setOnReady(() -> {
            String total = formatTime(media.getDuration());
            tiempoLabel.setText("00:00 / " + total);
        });

        mediaPlayer.play();
    }
}
