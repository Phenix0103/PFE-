package com.vermeg.risk.auth;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVWriter;
import com.vermeg.risk.entities.FinancialProfile;
import com.vermeg.risk.entities.SocialUser;
import com.vermeg.risk.repositories.FinancialProfileRepository;
import com.vermeg.risk.repositories.ClientRepository;
import com.vermeg.risk.services.ClientService;
import com.vermeg.risk.services.FinancialProfileService;
import com.vermeg.risk.services.PredictService;
import lombok.RequiredArgsConstructor;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import com.vermeg.risk.entities.Client;
import com.vermeg.risk.config.JwtService;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
@CrossOrigin(origins = "*", maxAge = 3600)


public class authController {
    @Value("${image.upload.directory}") // Inject the directory path from application.properties
    private String imageUploadDirectory;
    @Autowired
    private final authServices service;
    @Autowired
    private final FinancialProfileService financialProfileService;
   /* @Autowired
    private final ClientService clientService;*/
    @Autowired

    private final ClientRepository UR;
    @Autowired
    private final JwtService js;
    @Autowired
    private final UserDetails ud ;
    @Autowired
    private final PredictService s ;
    @Autowired

    private final FinancialProfileRepository financialProfileRepository;
    @Autowired

    private final ClientRepository clientRepository;
    // Assuming you have a FinancialProfileRepository
    private static final Logger logger = LoggerFactory.getLogger(authController.class);

    /*@PostMapping("/login")
    public ResponseEntity<String> login(@RequestBody authRequest loginRequest) {
        try {
            // Perform authentication
            String token = service.login(loginRequest.getEmail(), loginRequest.getPwd_user());

            // Return the JWT token in the response
            return ResponseEntity.ok(token);
        } catch (AuthenticationException e) {
            // Handle authentication errors
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Invalid email or password");
        }
    } */


    @PostMapping("/login")
    public ResponseEntity<authResponse> login(@RequestBody authRequest loginRequest) {
        try {
            // Perform authentication
            String token = service.login(loginRequest.getEmail(), loginRequest.getPwd_user());
            Client authenticatedUser = s.getUserByEmail(loginRequest.getEmail());

            // Create the AuthResponse object and populate its fields
            authResponse response = new authResponse();
            response.setToken(token);
            response.setWelcome("Welcome, " + authenticatedUser.getFirstname());
            response.setMessage("Authentication successful");
            response.setRole(authenticatedUser.getRole());
            response.setFirstname(authenticatedUser.getFirstname());
            response.setCin(authenticatedUser.getCin());
            response.setLastname(authenticatedUser.getLastname());
            response.setId_user(Math.toIntExact(authenticatedUser.getId()));


            // Return the AuthResponse object
            return ResponseEntity.ok(response);
        } catch (AuthenticationException e) {
            // Handle authentication errors
            authResponse errorResponse = new authResponse();
            errorResponse.setErrorMessage("Invalid email or password");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }
    }



    @PostMapping("/addClient")
    public ResponseEntity<String> addClient(
            @RequestParam(value="name") String clientName,
            @RequestParam(value = "firstname") String clientFirstName,
            @RequestParam(value = "cin") String clientCIN,
            @RequestParam(value = "address") String clientAddress,
            @RequestParam(value = "phonenumber") String clientPhoneNumber,
            @RequestParam(value = "email") String clientEmail,
            @RequestParam(value = "pwd") String clientPassword,
            @RequestParam(value = "image", required = false) MultipartFile imageFile
    ) {
        try {
            Client newClient = new Client();
            newClient.setName(clientName);
            newClient.setFirstname(clientFirstName);
            newClient.setCin(clientCIN);
            newClient.setAddress(clientAddress);
            newClient.setEmail(clientEmail);
            newClient.setPwd_user(clientPassword);

            if (imageFile != null && !imageFile.isEmpty()) {
                byte[] imageBytes = imageFile.getBytes();
                newClient.setImage(imageBytes); // Set the image data as byte array

                // Save the image to the specified directory
                String imagePath = imageUploadDirectory + "/" + imageFile.getOriginalFilename();
                Files.write(Paths.get(imagePath), imageBytes);
            }


            UR.save(newClient);

            return ResponseEntity.ok("Client added successfully");
        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Error adding client");
        }
    }
  /*  @PostMapping("/update-financial-profile/{clientId}")
    public ResponseEntity<Client> updateFinancialProfile(@PathVariable Long clientId, @RequestBody FinancialProfile financialProfile) {
        Client updatedClient = clientService.updateClientFinancialProfile(clientId, financialProfile);
        return new ResponseEntity<>(updatedClient, HttpStatus.OK);
    }
*/
    @GetMapping("/profile/{clientId}")
    public ResponseEntity<FinancialProfile> getFinancialProfileByClientId(@PathVariable Long clientId) {
        Optional<FinancialProfile> financialProfile = financialProfileService.getFinancialProfileByClientId(clientId);
        return financialProfile.map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/verify-code")
    public ResponseEntity<?> verifyUserCode(@RequestBody Map<String, String> codeMap) {
        String code = codeMap.get("code");
        try {
            service.verifyUserCode(code);
            return ResponseEntity.ok().build();
        } catch (IllegalArgumentException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An internal error occurred");
        }
    }

    @PostMapping(value = "/register", consumes = "multipart/form-data")
    public ResponseEntity<ApiResponse> register(
            @RequestParam("registerRequest") String registerRequestJson,
            @RequestParam("profile_image") MultipartFile profileImage) {

        try {
            // Convert JSON string to Java object
            ObjectMapper objectMapper = new ObjectMapper();
            registerRequest registerRequestObj = objectMapper.readValue(registerRequestJson, registerRequest.class);

            String email = registerRequestObj.getEmail();
            String password = registerRequestObj.getPassword();
            String firstname = registerRequestObj.getFirstname();
            String lastname = registerRequestObj.getLastname();
            String cin = registerRequestObj.getCin();
            String address = registerRequestObj.getAddress();
            String phonenumber = registerRequestObj.getPhonenumber();
            byte[] imageBytes = profileImage.getBytes();

            if (UR.existsByEmail(email)) {
                return new ResponseEntity<>(new ApiResponse("User with this email already exists"), HttpStatus.BAD_REQUEST);
            }

            service.register(email, password, firstname, lastname, cin, address, phonenumber,imageBytes);

            return ResponseEntity.ok(new ApiResponse("User registered successfully!"));

        } catch (IllegalArgumentException e) {
            return new ResponseEntity<>(new ApiResponse(e.getMessage()), HttpStatus.BAD_REQUEST);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    @DeleteMapping("/logout/{id}")
    public ResponseEntity<Map<String, String>> logout(@PathVariable Long id) {
        try {
            // Perform logout for the user with the given id
            service.logout(id);

            // Create a map for the response
            Map<String, String> response = new HashMap<>();
            response.put("status", "success");
            response.put("message", "User logged out successfully!");

            // Return a success message in the response
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            // Handle logout errors
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("status", "error");
            errorResponse.put("message", e.getMessage());

            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(errorResponse);
        }
    }
    @PostMapping("/validate-token")
    public ResponseEntity<?> validateToken(@RequestBody Map<String, String> payload) {
        String token = payload.get("token");
        UserDetails userDetails = loadUserByToken(token);
        boolean isValid = js.isTokenValid(token, userDetails);
        String role = userDetails.getAuthorities().stream()
                .findFirst()
                .orElseThrow(() -> new UsernameNotFoundException("User has no roles assigned"))
                .getAuthority();
        Map<String, Object> response = new HashMap<>();
        response.put("isValid", isValid);
        response.put("role", role);
        return ResponseEntity.ok(response);
    }
    public UserDetails loadUserByToken(String token) throws UsernameNotFoundException {
        Client user = UR.findByToken(token)
                .orElseThrow(() -> new UsernameNotFoundException("token not found with email: " + token));

        return user;
    }

    @PostMapping("/forgot-password")
    public ResponseEntity<String> forgotPassword(@RequestBody String email) {
        try {
            s.sendPasswordResetEmail(email);
            return ResponseEntity.ok("Password reset email sent!");
        } catch (UsernameNotFoundException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }

    @PostMapping("/reset-password")
    public ResponseEntity<String> resetPassword(@RequestParam String token, @RequestBody String newPassword) {
        try {
            s.resetPassword(token, newPassword);
            return ResponseEntity.ok("Password reset successful!");
        } catch (IllegalArgumentException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }
    @PostMapping("/google-login")
    public ResponseEntity<authResponse> googleLogin(@RequestBody SocialUser socialUser) {
        logger.info("Received Social User: {}", socialUser.toString());

        // Extract email and check if this user already exists in your DB
        Optional<Client> userOptional = UR.findByEmail(socialUser.getEmail());

        // Define a user reference outside the conditional check
        Client user;
        if (!userOptional.isPresent()) {
            user = new Client();
            user.setEmail(socialUser.getEmail());
            user.setFirstname(socialUser.getFirstName());
            user.setLastname(socialUser.getLastName());
            user.setName(socialUser.getName());
        } else {
            user = userOptional.get();
        }

        // This line updates the token for both new and existing users
        user.setToken(socialUser.getIdToken()); // Set idToken from SocialUser to token in Client

        user = UR.save(user); // Save and get the entity back which should have the ID populated

        // Generate a JWT token for this user
        String token = js.generateToken(user);

        authResponse response = new authResponse();
        response.setToken(token);
        response.setId_user(Math.toIntExact(user.getId())); // Set the id_user with the ID from the saved user

        return ResponseEntity.ok(response);
    }
    @Transactional
    public void someServiceOrDAOFunction(Client client) {
        UR.save(client);
    }
    @PutMapping("/update/{id}")
    public ResponseEntity<?> updateClient(
            @PathVariable Long id,
            @RequestParam("pwd_user") String chosenPwd,
            @RequestParam("profile_image") MultipartFile profileImage) {

        try {
            // Convert MultipartFile to byte array
            byte[] imageBytes = profileImage.getBytes();

            // Call the service method
            service.updateClient(id, chosenPwd, imageBytes);

            // Return a success response
            return ResponseEntity.ok().body(Map.of("message", "Client mis à jour avec succès"));
        } catch (Exception e) {
            // Handle any exceptions
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

        @GetMapping("/hasPasswordAndImage/{clientId}")
        public ResponseEntity<Boolean> hasPasswordAndImage(@PathVariable Long clientId) {
            boolean result = service.hasPasswordAndImage(clientId);
            return ResponseEntity.ok(result);
        }
    }






