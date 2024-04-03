package com.vermeg.risk.entities;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import javax.persistence.*;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

@Data
@Builder
@Entity
@Table
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class Client implements Serializable, UserDetails {
    private String verificationCode;
    private boolean verified = false;
    private LocalDateTime verificationCodeExpiry;

    @JsonProperty("id")
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @JsonProperty("lastName")
    private String lastname;

    @JsonProperty("firstName")
    private String firstname;

    @JsonProperty("name")
    private String name;

    @JsonProperty("cin")
    @Column(unique = true) // Add unique constraint
    private String cin;

    @JsonProperty("address")
    private String address;

    @JsonProperty("email")
    @NotBlank(message = "Email is required")
    @Email(message = "Invalid email format")
    @Column(unique = true) // Add unique constraint
    private String email;

    @JsonProperty("pwd_user")
    private String pwd_user;

    @JsonProperty("role")
    @Enumerated(EnumType.STRING)
    private UserEnum role;

    @JsonProperty("idToken")
    @Column(columnDefinition = "TEXT")
    private String token;

    @Lob
    private byte[] image; // Store image as base64 encoded string

    @OneToOne(cascade = CascadeType.ALL, mappedBy = "client")
    private FinancialProfile financialProfile;

    public void setImageData(byte[] imageData) {
        this.image = imageData;
    }

    public byte[] getImageData() {
        return image;
    }


    @Override
    @JsonIgnore
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return Collections.singletonList(new SimpleGrantedAuthority(role.name()));
    }

    @Override
    public String getPassword() {
        return pwd_user;
    }


    @Override
    public String getUsername() {
        return email;
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }



    public void setLastname(String lastname) {
        this.lastname = lastname;
        this.name = this.lastname;
    }




}


