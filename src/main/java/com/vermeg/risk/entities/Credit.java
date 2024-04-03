package com.vermeg.risk.entities;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import javax.validation.constraints.*;

@Entity
@Table
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Credit {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;

    @NotNull(message = "L'âge ne peut pas être nul.")
    @Min(value = 25, message = "L'âge doit être au moins 18.")
    @Max(value = 59, message = "L'âge doit être au maximum 100.")
    private Integer age;

    @NotNull(message = "Le niveau d'éducation ne peut pas être nul.")
    @Min(value = 1, message = "Le niveau d'éducation est invalide.")
    @Max(value = 5, message = "Le niveau d'éducation est invalide.")
    private Integer ed;

    @NotNull(message = "Le nombre d'années d'emploi ne peut pas être nul.")
    private Integer employ;

    @NotNull(message = "L'adresse est requise.")
    @Size(max = 255, message = "L'adresse est trop longue.")
    private String address;

    @NotNull(message = "Le revenu ne peut pas être nul.")
    @DecimalMin(value = "0.0", inclusive = false, message = "Le revenu doit être positif.")
    private Double income;

    @DecimalMin(value = "0.0", inclusive = false, message = "Le ratio d'endettement doit être positif.")
    private Double debtinc;

    @DecimalMin(value = "0.0", inclusive = false, message = "La dette de crédit doit être positif.")
    private Double creddebt;

    @DecimalMin(value = "0.0", inclusive = false, message = "L'autre dette doit être positif.")
    private Double othdebt;

    @NotNull(message = "L'indicateur de défaut est requis.")
    @Min(value = 0, message = "L'indicateur de défaut est invalide.")
    @Max(value = 1, message = "L'indicateur de défaut est invalide.")
    private Integer defaultIndicator;
}
