package com.vermeg.risk.auth;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)

public class registerRequest {
    private String email;
    private String password;
    private String firstname;
    private String lastname;
    private String cin;
    private String address;
    private String phonenumber;
    private String profileImage;
// getters and setters pour ce champ

}
