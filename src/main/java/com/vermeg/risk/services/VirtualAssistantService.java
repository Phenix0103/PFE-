package com.vermeg.risk.services;


import com.vermeg.risk.entities.FinancialProfile;
import com.vermeg.risk.entities.Client;
import com.vermeg.risk.repositories.ClientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class VirtualAssistantService {

    private final ClientRepository cr;

    @Autowired
    public VirtualAssistantService(ClientRepository cr) {
        this.cr = cr;
    }

    public String processClientQuestion(String userQuestion) {
        if (userQuestion.contains("profil financier")) {
            // Extract the customer's name from the userQuestion (using simple string manipulation)
            String customerName = extractCustomerName(userQuestion);

            if (customerName != null) {
                // Get the financial profile of the customer
                FinancialProfile financialProfile = getFinancialProfile(customerName);

                if (financialProfile != null) {
                    // Create a personalized response based on the customer's financial profile
                    String response = "Votre profil financier actuel :\n" +
                            "Age : " + financialProfile.getAge() + "\n" +
                            "Sexe : " + financialProfile.getGender() + "\n" +
                            "Emploi : " + financialProfile.getJob() + "\n" +
                            "Logement : " + financialProfile.getHousing() + "\n" +
                            "Comptes épargne : " + financialProfile.getSavingAccounts() + "\n" +
                    "Historique de credit : " + financialProfile.getCreditHistory() + "\n" +
                    "Montant : " + financialProfile.getCreditAmount() + "\n" +
                    "Durée : " + financialProfile.getDuration() + "\n" +
                    "Objectif : " + financialProfile.getPurpose() + "\n" +
                    "Risque : " + financialProfile.getRisk()  ;

                    // You can add more personalized information based on the financialProfile object
                    // ...

                    return response;
                } else {
                    return "Nous n'avons pas trouvé de profil financier pour le client " + customerName + ". Veuillez vérifier le nom ou fournir des informations financières.";
                }
            } else {
                return "Veuillez fournir votre nom complet pour accéder à votre profil financier.";
            }
        } else {
            // For other questions, process them as usual
            return getChatbotResponse(userQuestion);
        }
    }

    private String extractCustomerName(String userQuestion) {
        String customerName = null;
        int startIndex = userQuestion.indexOf("Je m'appelle");
        if (startIndex != -1) {
            startIndex += "Je m'appelle".length() + 1;
            int endIndex = userQuestion.indexOf(" ", startIndex);
            if (endIndex != -1) {
                customerName = userQuestion.substring(startIndex, endIndex);
            } else {
                customerName = userQuestion.substring(startIndex);
            }
        }
        return customerName;
    }

    // Define a method to execute the Python script
    private String getChatbotResponse(String userQuestion) {
        // Your existing code for getting chatbot response goes here
        // ...
        return userQuestion;
    }

    public FinancialProfile getFinancialProfile(String customerName) {
        Client client = cr.findByName(customerName);
        if (client != null) {
            return client.getFinancialProfile();
        } else {
            return null;
        }
    }
}
