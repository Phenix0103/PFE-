package com.vermeg.risk.entities;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Data
@Builder
public class PredictionResult {

    @JsonProperty("Prediction")
    private Double prediction;

    @JsonProperty("Probabilité de défaut de paiement")
    private String probabilitéDeDéfautDePaiement;

    @JsonProperty("Probabilité de remboursement du prêt")
    private String probabilitéDeRemboursementDuPrêt;

    @JsonProperty("Message")
    private String message;

    @JsonProperty("Observations")
    private List<String> observations;

}
