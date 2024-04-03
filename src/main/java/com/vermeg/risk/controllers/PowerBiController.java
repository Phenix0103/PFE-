package com.vermeg.risk.controllers;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/powerbi")
public class PowerBiController {

    @GetMapping("/report-url")
    public String getPowerBiReportUrl() {
        // Remplacez l'URL par celle de votre rapport Power BI
        return "https://app.powerbi.com/reportEmbed?reportId=180c3917-2f0b-4afb-8d0f-23a7b66fc5ed&autoAuth=true&ctid=513486ec-6643-4f17-a508-76478311be42";
    }
}
