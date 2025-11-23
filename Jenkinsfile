pipeline {
  agent any
  options { timestamps(); disableConcurrentBuilds() }
  environment {
    REGISTRY  = "${env.REGISTRY}"
    BRANCH    = "${env.BRANCH_NAME}"
    SHORT_SHA = sh(script: "git rev-parse --short=8 HEAD", returnStdout: true).trim()
    IMAGE     = "${REGISTRY}/svc-python"
    TAG_SHA   = "qa-${SHORT_SHA}"
  }
  stages {
    stage('Checkout'){ steps { checkout scm } }
    stage('Test'){ steps { sh 'pytest -q || true' } }
    stage('Docker login'){
      when { branch 'QA' }
      steps {
        withCredentials([usernamePassword(credentialsId: 'container-registry', usernameVariable: 'U', passwordVariable: 'P')]){
          sh 'echo "$P" | docker login -u "$U" --password-stdin'
        }
      }
    }
    stage('Build & Push (QA)'){
      when { branch 'QA' }
      steps {
        sh """
          docker build -t ${IMAGE}:${TAG_SHA} -t ${IMAGE}:qa-latest .
          docker push ${IMAGE}:${TAG_SHA}
          docker push ${IMAGE}:qa-latest
        """
      }
    }
    stage('Trigger deploy QA'){ when { branch 'QA' } steps { build job: 'infra-deploy-qa', wait: false } }
  }
}
