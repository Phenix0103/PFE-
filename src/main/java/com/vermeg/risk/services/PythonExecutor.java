package com.vermeg.risk.services;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PythonExecutor {
    public static void executePythonScript() {
        try {
            // Spécifiez le chemin vers votre script Python
            String pythonScriptPath = "C:\\Users\\ASUS TUF I5\\Desktop\\Vermeg\\Scrapping2\\new.py";

            // Créez un processus Python
            Process process = new ProcessBuilder("python", pythonScriptPath).start();

            // Capturez la sortie du script Python
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            // Attendez que le processus Python se termine
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                System.out.println("Le script Python s'est terminé avec succès.");
            } else {
                System.err.println("Le script Python s'est terminé avec une erreur.");
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
