pipeline {
    agent any
    tools {
        maven 'M2_HOME'
    }
    stages {
        stage('GIT') {
            steps {
                git branch: 'master',
                url: 'https://github.com/Phenix0103/PFE-.git'
            }
        }
       
        stage('Install dependencies') {
            steps {
                // Utilisez pip pour installer les dépendances à partir de requirements.txt
                sh 'pip install -r Requirements.txt'
            }
        }
        // Supposons que l'installation de dépendances Python et l'entraînement du modèle sont gérés par General.py
        stage('Évaluation et Entrainement du Modèle') {
            steps {
                script {
                    // Exécutez votre script Python qui devrait gérer l'installation de dépendances
                    sh 'python3 ROE.py'
                    sh 'python3 ROA.py'
                    sh 'python3 Ratio.py'
                    sh 'python3 Credit.py'
                    sh 'python3 Pret.py'
                    sh 'python3 PIB.py'
                    sh 'python3 levier.py'
                    sh 'python3 Gestion.py'
                    sh 'python3 Distribution.py'
                    //sh 'python3 Change.py' 
                }
            }
        }
 
        
    
        

      
       
        stage('Docker') {
            steps {
                script {
                    // Build the Docker image with Jenkins BUILD_NUMBER as the version
                    sh 'docker build -t kaddemimage:v${BUILD_NUMBER} -f Dockerfile ./'
                    
                    // Tagging the Docker image for Docker Hub
                    sh 'docker tag kaddemimage:v${BUILD_NUMBER} ceceyphoenix/pfecyrine:v${BUILD_NUMBER}'

                    // Login to Docker Hub (Ensure Docker Hub credentials are configured in Jenkins)
                    // The 'dockerhubcredentials' should be the ID of your Docker Hub credentials stored in Jenkins
                    sh 'docker login --username ceceyphoenix --password Princesseflora1'
                    
                    // Push the Docker image to Docker Hub
                    sh 'docker push ceceyphoenix/pfecyrine:v${BUILD_NUMBER}'
                    
                    // Run Docker Compose
                    sh "IMAGE_VERSION=v${BUILD_NUMBER} docker compose up -d"
                }
            }
        }
        
        stage('Grafana') {
            steps {
                sh 'docker compose up -d'
            }
        }
    } // Fin des stages
    
    /*post {
        success {
            mail to: 'chouchanecyrine@gmail.com',
                 subject: "SUCCESS: Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "The pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER} completed successfully."
        }
        failure {
            mail to: 'chouchanecyrine@gmail.com',
                 subject: "FAILURE: Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "The pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER} failed."
        }
        unstable {
            mail to: 'chouchanecyrine@gmail.com',
                 subject: "UNSTABLE: Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "The pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER} is unstable."
        }
        aborted {
            mail to: 'chouchanecyrine@gmail.com',
                 subject: "ABORTED: Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "The pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER} was aborted."
        }
    }*/
} // Fin du pipeline
