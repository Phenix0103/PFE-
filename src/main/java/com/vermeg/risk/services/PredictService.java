package com.vermeg.risk.services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvValidationException;
import com.vermeg.risk.auth.authServices;
import com.vermeg.risk.entities.Credit;
import com.vermeg.risk.entities.PredictionResult;
import com.vermeg.risk.entities.Client;
import com.vermeg.risk.repositories.ClientRepository;
import com.vermeg.risk.repositories.PredictionRepository;
import com.vermeg.risk.repositories.ClientRepository;
import lombok.AllArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import com.fasterxml.jackson.core.type.TypeReference;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

@AllArgsConstructor
@Service
public class PredictService {

    private final RestTemplate restTemplate = new RestTemplate();
    @Autowired
    private final PredictionRepository predictionRepository;
    @Autowired
    private final ClientRepository userRepository;
    @Autowired
    authServices emailService ;

    @Autowired
    private JavaMailSender mailSender;
    @Autowired
    private static final Logger logger = LoggerFactory.getLogger(PredictService.class); // replace 'YourClassName' with the name of your class
    @Autowired
    private final PasswordEncoder passwordEncoder;

    public PredictionResult predict(Credit credit) throws JsonProcessingException {
        String flaskApiUrl = "http://localhost:5001/predict";

        Map<String, Object> data = new HashMap<>();
        data.put("age", credit.getAge());
        data.put("ed", credit.getEd());
        data.put("employ", credit.getEmploy());
        data.put("address", credit.getAddress());
        data.put("income", credit.getIncome());
        data.put("debtinc", credit.getDebtinc());
        data.put("creddebt", credit.getCreddebt());
        data.put("othdebt", credit.getOthdebt());

        PredictionResult flaskResponse = restTemplate.postForObject(flaskApiUrl, data, PredictionResult.class);

        if (flaskResponse != null && flaskResponse.getPrediction() != null) {
            credit.setDefaultIndicator(flaskResponse.getPrediction().intValue());
        }

        // Add data to CSV
        List<String[]> csvData = new ArrayList<>();
        csvData.add(new String[]{
                String.valueOf(credit.getAge()),
                String.valueOf(credit.getEd()),
                String.valueOf(credit.getEmploy()),
                String.valueOf(credit.getAddress()),
                credit.getIncome() % 1.0 == 0 ? String.valueOf((int) Math.round(credit.getIncome())) : String.valueOf(credit.getIncome()),
                credit.getDebtinc() % 1.0 == 0 ? String.valueOf((int) Math.round(credit.getDebtinc())) : String.valueOf(credit.getDebtinc()),
                credit.getCreddebt() % 1.0 == 0 ? String.valueOf((int) Math.round(credit.getCreddebt())) : String.valueOf(credit.getCreddebt()),
                credit.getOthdebt() % 1.0 == 0 ? String.valueOf((int) Math.round(credit.getOthdebt())) : String.valueOf(credit.getOthdebt()),
                String.valueOf(credit.getDefaultIndicator())
        });

        addDataToCSV(csvData, "C:/Users/ahmed/OneDrive/Documents/NetBeansProjects/bankloans.csv");

        return flaskResponse;
    }


    public void addDataToCSV(List<String[]> newData, String csvFilePath) {
        List<String[]> existingData = new ArrayList<>();

        // Étape 1: Lire le contenu actuel du fichier CSV
        try (CSVReader reader = new CSVReader(new FileReader(csvFilePath))) {
            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                existingData.add(nextLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (CsvValidationException e) {
            throw new RuntimeException(e);
        }

        // Étape 2: Vérifier si la ligne que vous essayez d'ajouter existe déjà dans le fichier
        boolean dataExists = false;
        for (String[] existingLine : existingData) {
            if (Arrays.equals(existingLine, newData.get(0))) { // Comparer la nouvelle ligne avec chaque ligne existante
                dataExists = true;
                break;
            }
        }

        // Étape 3: Si elle n'existe pas, ajoutez-la
        if (!dataExists) {
            try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath, true), ',', CSVWriter.NO_QUOTE_CHARACTER, CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END)) {
                writer.writeAll(newData);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    public void sendPasswordResetEmail(String email) {
        String actualEmail = email
                .replace("{", "")
                .replace("}", "")
                .replace("\"email\":\"", "")
                .replace("\"", "")
                .trim();
        // Log the extracted email for debugging
        System.out.println("Extracted Email: " + actualEmail);


        Optional<Client> userOptional = userRepository.findByEmail(actualEmail);

        System.out.println("Looking for user with email: " + actualEmail); // logging
        if (userOptional.isEmpty()) {
            System.out.println("User not found!"); // logging
            throw new UsernameNotFoundException("No user found with this email");
        }

        Client user = userOptional.get();
        String token = UUID.randomUUID().toString();
        user.setToken(token);
        userRepository.save(user);

        String resetUrl = "http://localhost:8081/auth/reset-password?token=" + token;
        emailService.sendResetPasswordEmail(actualEmail, resetUrl);
    }
    public void resetPassword(String token, String rawPasswordInput) {
        String actualPassword = extractPasswordFromRawInput(rawPasswordInput);

        Optional<Client> userOptional = userRepository.findByToken(token);
        if (userOptional.isEmpty()) {
            throw new IllegalArgumentException("Invalid token");
        }

        Client user = userOptional.get();

        // Log the extracted password (for debugging purposes)
        logger.info("Extracted new password: {}", actualPassword);

        user.setPwd_user(passwordEncoder.encode(actualPassword));  // Hashing the password before saving
        userRepository.save(user);
    }


    public String extractPasswordFromRawInput(String rawInput) {
        return rawInput
                .replace("{", "")
                .replace("}", "")
                .replace("\"newPassword\":\"", "")
                .replace("\"", "")
                .trim();
    }
    public Client getUserByEmail(String email) {
        return userRepository.findByEmail1(email);
    }

}