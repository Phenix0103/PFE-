pipeline {
    agent any
    tools {
        maven 'M2_HOME'
    }
    stages {
        stage('GIT') {
            steps {
                git branch: 'Main',
                url: 'https://github.com/Phenix0103/PFE-.git'
            }
        }
        stage('MVN CLEAN') {
            steps {
                sh 'mvn clean';
            }
        }
        stage('MVN COMPILE') {
            steps {
                sh 'mvn compile';
            }
        }
        stage('MVN SONARQUBE') {
            steps {
                sh 'mvn sonar:sonar -Dsonar.login=admin -Dsonar.password=cyrine -Dmaven.test.skip=true';
            }
        }
        stage('MOCKITO'){
            steps {
                 sh 'mvn test';
            }
        }
        stage('NEXUS'){
            steps {
                 sh 'mvn deploy';
            }
        }
        // Ajoutez ici les étapes pour l'entraînement et l'évaluation du modèle
        stage('Entraînement du Modèle') {
            steps {
                script {
                    // Assurez-vous que le script d'entraînement est bien configuré et accessible
                    sh 'python train_model.py'
                }
            }
        }
        stage('Évaluation du Modèle') {
            steps {
                script {
                    // Exécutez votre script d'évaluation, assurez-vous qu'il imprime des métriques de performance claires
                    sh 'python evaluate_model.py'
                }
            }
        }
        stage('Docker') {
            steps {
                script {
                    sh 'docker build -t kaddemimage:v${BUILD_NUMBER} -f Dockerfile ./'
                    sh 'docker tag kaddemimage:v${BUILD_NUMBER} ceceyphoenix/projetdevops:v${BUILD_NUMBER}'
                    sh 'docker login --username ceceyphoenix --password Princesseflora1'
                    sh 'docker push ceceyphoenix/projetdevops:v${BUILD_NUMBER}'
                    sh "IMAGE_VERSION=v${BUILD_NUMBER} docker compose up -d"
                }
            }
        }
        stage('Grafana') {
            steps {
                sh 'docker compose up -d'
            }
        }
    }
    post {
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
    }
}
