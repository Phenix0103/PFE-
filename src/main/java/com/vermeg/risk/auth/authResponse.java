package com.vermeg.risk.auth;

import com.vermeg.risk.entities.UserEnum;
import lombok.*;
import com.vermeg.risk.entities.Client;


import javax.persistence.Table;
import java.util.List;

@Data
@Builder
@Table
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor

public class authResponse {
    private String token;
    private String welcome;
    private int id_user; // Add this field

    private String message;
    private String errorMessage;
    private UserEnum role ;
    private String firstname ;
    private String lastname ;
    private String cin ;
    public String getFullName() {
        return firstname + " " + lastname;
    }
    // getter and setter methods
}
