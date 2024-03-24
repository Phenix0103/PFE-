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
/*stage('MVN SONARQUBE') {
            steps {
                sh 'mvn sonar:sonar -Dsonar.login=admin -Dsonar.password=cyrine -Dmaven.test.skip=true';
            }
        }

stage('NEXUS'){
            steps {
                 sh 'mvn deploy';
            }
        }
       */
    }
  /*  post {
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
}
