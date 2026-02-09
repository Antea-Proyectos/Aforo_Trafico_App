/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.aforotraficoapp;

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
        FXMLLoader loader= new FXMLLoader(getClass().getResource("/MainView.fxml"));
        Scene scene= new Scene(loader.load());
        stage.setScene(scene);
        stage.setTitle("Aforo de Tráfico");
        stage.show();
    }
    public static void main(String[] args) {
        launch();
    }
}
