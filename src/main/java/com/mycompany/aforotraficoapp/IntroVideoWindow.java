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

/**
 *
 * @author nereatrillo
 */
public class IntroVideoWindow {

    private final Runnable onClose;

    public IntroVideoWindow(Runnable onClose) {
        this.onClose = onClose;
    }

    public void mostrar() {
        Stage stage = new Stage();
        stage.setTitle("Bienvenida");
        stage.setOnCloseRequest(e -> {
            stage.close();
            onClose.run();
        });

        // Ruta del vídeo dentro del proyecto
        String videoPath = getClass().getResource("/videos/info.mp4").toExternalForm();

        Media media = new Media(videoPath);
        MediaPlayer mediaPlayer = new MediaPlayer(media);
        MediaView mediaView = new MediaView(mediaPlayer);

        BorderPane root = new BorderPane(mediaView);
        Scene scene = new Scene(root, 1280, 720);

        stage.setScene(scene);
        stage.setFullScreen(true);
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.show();
        
        mediaPlayer.setOnEndOfMedia(() -> {
            Preferences.userNodeForPackage(Main.class).putBoolean("intro_mostrada", true);
            mediaPlayer.stop();
            stage.close();
            onClose.run();
        });
        mediaPlayer.play();
    }
}
