package com.vermeg.risk.controllers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.util.MultiValueMap;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


@CrossOrigin(origins = {"*"}, maxAge = 3600L)
@RestController
@RequestMapping("/api")
public class ChatbotController {

    @Value("${python.api.url}")  // This should be configured in your application.properties or YAML file
    private String pythonApiUrl;

    private final RestTemplate restTemplate;

    @Autowired
    public ChatbotController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @PostMapping("/chat")
    public ResponseEntity<String> chatWithPython(@RequestBody Map<String, String> request) {
        String userMessage = request.get("message");

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("message", userMessage);

        HttpEntity<Map<String, String>> entity = new HttpEntity<>(requestBody, headers);

        ResponseEntity<String> response = restTemplate.exchange(
                pythonApiUrl + "/chat",  // The Python API endpoint
                HttpMethod.POST,
                entity,
                String.class
        );

        if (response.getStatusCode() == HttpStatus.OK) {
            return ResponseEntity.ok(response.getBody());
        } else {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An error occurred.");
        }
    }

    @PostMapping("/upload")
    public ResponseEntity<String> uploadFileToPython(@RequestParam("csv-file") MultipartFile file) throws IOException {
        System.out.println("Nom du fichier : " + file.getOriginalFilename());
        System.out.println("Taille du fichier : " + file.getSize());

        if (!file.isEmpty()) {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("csv-file", resource);

            HttpEntity<MultiValueMap<String, Object>> entity = new HttpEntity<>(body, headers);

            try {
                ResponseEntity<String> response = restTemplate.exchange(
                        pythonApiUrl + "/upload",
                        HttpMethod.POST,
                        entity,
                        String.class
                );
                return ResponseEntity.status(response.getStatusCode()).body(response.getBody());
            } catch (Exception e) {
                System.err.println("Erreur lors de l'envoi du fichier au serveur Flask : " + e.getMessage());
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Une erreur s'est produite lors de l'envoi du fichier.");
            }

        } else {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Pas de fichier téléchargé.");
        }
    }

    @PostMapping("/reset-chatbot")
    public ResponseEntity<String> resetChatbot() {
        // Send a request to the Python Flask endpoint to reset the chatbot
        ResponseEntity<String> response = restTemplate.postForEntity(
                pythonApiUrl + "/reset-chatbot",
                null,
                String.class
        );

        if (response.getStatusCode() == HttpStatus.OK) {
            return ResponseEntity.ok(response.getBody());
        } else {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An error occurred.");
        }
    }
}