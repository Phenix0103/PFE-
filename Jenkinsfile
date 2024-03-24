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
