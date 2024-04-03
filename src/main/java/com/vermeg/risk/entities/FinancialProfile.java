package com.vermeg.risk.entities;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;

import javax.persistence.*;

@Entity
@Getter
@Setter
@NoArgsConstructor
public class FinancialProfile {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private int age;
    private double gender;
    private double job;
    private double housing;
    private double savingAccounts;
    private double creditHistory;
    private double creditAmount;
    private double duration;
    private double purpose;
    private double risk;

    @JsonIgnore // Cette propriété ne sera pas incluse dans la sérialisation JSON
    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "client_id")
    private Client client;

}
