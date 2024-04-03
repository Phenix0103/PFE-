package com.vermeg.risk.services;

import com.opencsv.CSVWriter;
import com.vermeg.risk.entities.FinancialProfile;
import com.vermeg.risk.repositories.FinancialProfileRepository;
import com.vermeg.risk.repositories.ClientRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.persistence.EntityNotFoundException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Optional;

@Service
public class FinancialProfileServiceImpl implements FinancialProfileService {

    private final FinancialProfileRepository financialProfileRepository;
    private final ClientRepository clientRepository;

    @Autowired
    public FinancialProfileServiceImpl(FinancialProfileRepository financialProfileRepository, ClientRepository clientRepository) {
        this.financialProfileRepository = financialProfileRepository;
        this.clientRepository = clientRepository;
    }

    @Override
    public FinancialProfile createFinancialProfile(FinancialProfile financialProfile) {
        return financialProfileRepository.save(financialProfile);
    }

    @Override
    public FinancialProfile updateFinancialProfile(Long id, FinancialProfile updatedFinancialProfile) {
        FinancialProfile existingFinancialProfile = financialProfileRepository.findById(id)
                .orElseThrow(() -> new EntityNotFoundException("Profil financier introuvable avec l'ID : " + id));

        // Update the necessary properties of the existing entity
        existingFinancialProfile.setAge(updatedFinancialProfile.getAge());
        existingFinancialProfile.setGender(updatedFinancialProfile.getGender());
        existingFinancialProfile.setJob(updatedFinancialProfile.getJob());
        existingFinancialProfile.setHousing(updatedFinancialProfile.getHousing());
        existingFinancialProfile.setSavingAccounts(updatedFinancialProfile.getSavingAccounts());
        existingFinancialProfile.setCreditHistory(updatedFinancialProfile.getCreditHistory());
        existingFinancialProfile.setCreditAmount(updatedFinancialProfile.getCreditAmount());
        existingFinancialProfile.setDuration(updatedFinancialProfile.getDuration());
        existingFinancialProfile.setPurpose(updatedFinancialProfile.getPurpose());
        existingFinancialProfile.setRisk(updatedFinancialProfile.getRisk());
        // Save the updated entity
        return financialProfileRepository.save(existingFinancialProfile);
    }

    @Override
    public FinancialProfile addFinancialProfile(FinancialProfile financialProfile) {
        // Calculate net worth based on provided values
        // Save the FinancialProfile to the database
        return financialProfileRepository.save(financialProfile);
    }



    public byte[] generateCSV(FinancialProfile financialProfile) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        CSVWriter csvWriter = new CSVWriter(new OutputStreamWriter(byteArrayOutputStream));

        // Replace column names and data with those of your financial model
        String[] header = { "Age", "Gender", "Job", "Housing", "SavingAccounts", "CreditHistory", "CreditAmount",
                "Duration", "Purpose", "Risk"};
        String[] data = {
                String.valueOf(financialProfile.getAge()),
                String.valueOf(financialProfile.getGender()),
                String.valueOf(financialProfile.getJob()),
                String.valueOf(financialProfile.getHousing()),
                String.valueOf(financialProfile.getSavingAccounts()),
                String.valueOf(financialProfile.getCreditHistory()),
                String.valueOf(financialProfile.getCreditAmount()),
                String.valueOf(financialProfile.getDuration()),
                String.valueOf(financialProfile.getPurpose()),
                String.valueOf(financialProfile.getRisk()),
        };

        csvWriter.writeNext(header);
        csvWriter.writeNext(data);

        csvWriter.close();
        return byteArrayOutputStream.toByteArray();
    }

    public Optional<FinancialProfile> getFinancialProfileByClientId(Long clientId) {
        return financialProfileRepository.findByClientId(clientId);
    }

    @Override
    public FinancialProfile createOrUpdateFinancialProfile(FinancialProfile financialProfile) {
        return financialProfileRepository.save(financialProfile);
    }

    @Override
    public FinancialProfile saveOrUpdateFinancialProfile(Long clientId, FinancialProfile financialProfile) {
        Optional<FinancialProfile> existingProfileOptional = financialProfileRepository.findById(clientId);
        if (existingProfileOptional.isPresent()) {
            FinancialProfile existingProfile = existingProfileOptional.get();
            existingProfile.setAge(financialProfile.getAge());
            existingProfile.setGender(financialProfile.getGender());
            existingProfile.setJob(financialProfile.getJob());
            existingProfile.setHousing(financialProfile.getHousing());
            existingProfile.setSavingAccounts(financialProfile.getSavingAccounts());
            existingProfile.setCreditHistory(financialProfile.getCreditHistory());
            existingProfile.setCreditAmount(financialProfile.getCreditAmount());
            existingProfile.setDuration(financialProfile.getDuration());
            existingProfile.setPurpose(financialProfile.getPurpose());
            existingProfile.setRisk(financialProfile.getRisk());

            return financialProfileRepository.save(existingProfile);
        } else {
            return financialProfileRepository.save(financialProfile);
        }
    }
}
