package com.mycompany.aforotraficoapp;

import java.util.prefs.Preferences;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;
/**
 *
 * @author nereatrillo
 */
public class Main extends Application {

    @Override
    public void start(Stage stage) throws Exception {

        Preferences prefs = Preferences.userNodeForPackage(Main.class);
        boolean introMostrada = prefs.getBoolean("intro_mostrada", false);
         
        if (!introMostrada) {
            // Mostrar ventana emergente con vídeo
            IntroVideoWindow intro = new IntroVideoWindow(() -> {
                prefs.putBoolean("intro_mostrada", true);
                mostrarVentanaPrincipal(stage);
            });
            intro.mostrar();
        } else {
            mostrarVentanaPrincipal(stage);
        }
    }

    private void mostrarVentanaPrincipal(Stage stage) {
        try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/MainView.fxml"));
            Scene scene = new Scene(loader.load());
            stage.setScene(scene);
            stage.setTitle("Aforo de Tráfico");
            stage.show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        launch();
    }
}
