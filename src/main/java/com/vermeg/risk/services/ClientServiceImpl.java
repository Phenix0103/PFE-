/*package com.vermeg.risk.services;

import com.vermeg.risk.entities.Client;
import com.vermeg.risk.entities.FinancialProfile;
import com.vermeg.risk.repositories.ClientRepository;
import com.vermeg.risk.repositories.FinancialProfileRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.persistence.EntityNotFoundException;
import javax.transaction.Transactional;
import java.util.Optional;

@Service
public class FinancialProfileService {

    private final FinancialProfileRepository financialProfileRepository;

    @Autowired
    public FinancialProfileService(FinancialProfileRepository financialProfileRepository) {
        this.financialProfileRepository = financialProfileRepository;
    }

    public FinancialProfile saveOrUpdateFinancialProfile(Long clientId, FinancialProfile financialProfile) {
        // Vérifiez d'abord si le client existe dans la table Financial Profile
        Optional<FinancialProfile> existingProfileOptional = financialProfileRepository.findByClientId(clientId);

        if (existingProfileOptional.isPresent()) {
            // Si le client existe, mettez à jour le profil financier existant
            FinancialProfile existingProfile = existingProfileOptional.get();
            updateExistingProfile(existingProfile, financialProfile);
            return financialProfileRepository.save(existingProfile);
        }

        // Si le client n'existe pas dans la table Financial Profile,
        // créez un nouveau profil financier avec l'ID du client
        financialProfile.setClient(new Client(clientId)); // Crée un client avec l'ID
        financialProfile.setNetWorth(calculateNetWorth(financialProfile));
        return financialProfileRepository.save(financialProfile);
    }

    private void updateExistingProfile(FinancialProfile existingProfile, FinancialProfile updatedProfile) {
        existingProfile.setIncome(updatedProfile.getIncome());
        existingProfile.setExpenses(updatedProfile.getExpenses());
        existingProfile.setAssets(updatedProfile.getAssets());
        existingProfile.setLiabilities(updatedProfile.getLiabilities());
        existingProfile.setCsv(updatedProfile.getCsv());
        existingProfile.setCsvFileName(updatedProfile.getCsvFileName());

        double netWorth = calculateNetWorth(existingProfile);
        existingProfile.setNetWorth(netWorth);
    }

    // Implémentez la logique de calcul du net worth ici
    private double calculateNetWorth(FinancialProfile financialProfile) {
        // Vous pouvez utiliser les propriétés de financialProfile telles que income, expenses, assets, liabilities, etc.
        // pour calculer le net worth en fonction de votre logique métier
        // Par exemple :
        double netWorth = financialProfile.getAssets() - financialProfile.getLiabilities();
        return netWorth;
    }
}*/
