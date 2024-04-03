package com.vermeg.risk.services;

import com.vermeg.risk.entities.FinancialProfile;

import java.io.IOException;
import java.util.Optional;

public interface FinancialProfileService{

    FinancialProfile updateFinancialProfile(Long id, FinancialProfile updatedFinancialProfile);

    FinancialProfile addFinancialProfile(FinancialProfile financialProfile);

    FinancialProfile createOrUpdateFinancialProfile(FinancialProfile financialProfile);

    FinancialProfile createFinancialProfile(FinancialProfile financialProfile);


    byte[] generateCSV(FinancialProfile financialProfile) throws IOException;

    Optional<FinancialProfile> getFinancialProfileByClientId(Long clientId);

    FinancialProfile saveOrUpdateFinancialProfile(Long clientId, FinancialProfile financialProfile);
}
