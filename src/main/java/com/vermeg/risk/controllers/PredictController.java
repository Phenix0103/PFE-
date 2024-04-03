package com.vermeg.risk.controllers;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.vermeg.risk.entities.Credit;
import com.vermeg.risk.entities.PredictionResult;
import com.vermeg.risk.services.PredictService;
import lombok.AllArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*", maxAge = 3600)
@RequiredArgsConstructor

public class PredictController {
    @Autowired
    private final PredictService predictService;


    @PostMapping("/loanApplications")
    public ResponseEntity<PredictionResult> createLoanApplication(@RequestBody Credit credit) throws JsonProcessingException {
        PredictionResult flaskResponse = predictService.predict(credit);
        return ResponseEntity.ok(flaskResponse);
    }
}