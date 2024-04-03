package com.vermeg.risk.controllers;

import com.vermeg.risk.entities.Client;
import com.vermeg.risk.entities.FinancialProfile;
;
import com.vermeg.risk.repositories.ClientRepository;
import com.vermeg.risk.services.FinancialProfileService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@RestController
@RequestMapping("/financial-profiles")
public class FinancialProfileController {
@Autowired
    private final FinancialProfileService financialProfileService;
@Autowired
    private final ClientRepository clientRepository;

    @Autowired
    public FinancialProfileController(FinancialProfileService financialProfileService, ClientRepository clientRepository) {
        this.financialProfileService = financialProfileService;
        this.clientRepository = clientRepository;
    }

    @PostMapping("/save-or-update/{clientId}")
    public ResponseEntity<?> saveOrUpdateFinancialProfile(@PathVariable Long clientId, @RequestBody FinancialProfile financialProfile) {
        try {
            // Fetch the Client entity by its ID
            Optional<Client> clientOptional = clientRepository.findById(clientId);

            if (clientOptional.isPresent()) {
                Client client = clientOptional.get();

                // Set the Client entity for the FinancialProfile
                financialProfile.setClient(client);

                // Save or update the FinancialProfile
                FinancialProfile updatedFinancialProfile = financialProfileService.saveOrUpdateFinancialProfile(clientId, financialProfile);

                return new ResponseEntity<>(updatedFinancialProfile, HttpStatus.OK);
            } else {
                return new ResponseEntity<>("Client with ID " + clientId + " not found.", HttpStatus.NOT_FOUND);
            }
        } catch (Exception e) {
            return new ResponseEntity<>("Erreur lors de la mise à jour du profil financier : " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
    @GetMapping("/profile/{clientId}")
    public ResponseEntity<FinancialProfile> getFinancialProfileByClientId(@PathVariable Long clientId) {
        Optional<FinancialProfile> financialProfile = financialProfileService.getFinancialProfileByClientId(clientId);
        return financialProfile.map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }
}