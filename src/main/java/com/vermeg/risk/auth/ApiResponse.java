package com.vermeg.risk.auth;

import lombok.*;

import javax.persistence.Table;

@Data
@Builder
@Table
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor

public class ApiResponse {
    private String message;

}
