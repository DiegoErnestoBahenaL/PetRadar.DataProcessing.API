pipeline {
    agent any

    options {
        // Evita builds en paralelo que se pisen el docker-compose
        disableConcurrentBuilds()
        timestamps()
    }

    environment {
        PROJECT_ROOT   = '/opt/petradar-qa'
        COMPOSE_FILE   = "${PROJECT_ROOT}/docker-compose.qa.yml"
        DOCKER_BUILDKIT = '0'
    }

    stages {
        stage('Solo rama QA') {
            when {
                expression { env.BRANCH_NAME == 'QA' }
            }
            steps {
                echo "Ejecutando pipeline QA para ${env.JOB_NAME} / ${env.BRANCH_NAME}"
            }
        }

        stage('Sync repos QA') {
            when { expression { env.BRANCH_NAME == 'QA' } }
            steps {
                sh '''
                    set -e

                    cd ${PROJECT_ROOT}/PetRadar.Web.API
                    git fetch origin
                    git checkout QA
                    git pull origin QA

                    cd ${PROJECT_ROOT}/PetRadar.DataProcessing.API
                    git fetch origin
                    git checkout QA
                    git pull origin QA

                    cd ${PROJECT_ROOT}/PetRadar.Web.UI
                    git fetch origin
                    git checkout QA
                    git pull origin QA
                '''
            }
        }

        stage('Build imágenes QA') {
            when { expression { env.BRANCH_NAME == 'QA' } }
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_ROOT}
                    DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker compose -f ${COMPOSE_FILE} build
                '''
            }
        }

        stage('Deploy stack QA') {
            when { expression { env.BRANCH_NAME == 'QA' } }
            steps {
                sh '''
                    set -e
                    cd ${PROJECT_ROOT}
                    DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker compose -f ${COMPOSE_FILE} up -d
                '''
            }
        }
    }
}
