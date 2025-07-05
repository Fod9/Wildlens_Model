pipeline {
    agent {
        node {
            label 'python-3-12'
        }
    }

    triggers {
        GenericTrigger(
         genericVariables: [
            [key: 'ref', value: '$.ref']
         ],
         genericHeaderVariables: [
            [key: 'X-GitHub-Event', regexpFilter: '']
         ],
    
         causeString: 'Triggered on $ref',
    
         token: 'wildlens_training',
    
         printContributedVariables: true,
         printPostContent: true,
    
         silentResponse: false,
    
         regexpFilterText: '$ref',
         regexpFilterExpression: 'refs/heads/' + 'master'
        )
    }

    environment {
        SHARED_MODEL_PATH = "/home/shared/Wildlens/models/multiclassifier/wildlens_multiclassifier.keras"
    }

    stages {
        stage('Install Dependencies') {
            steps {
                sh '''
                        rm -rf venv || true

                        python3 -m venv venv

                        . venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                        . venv/bin/activate

                        python3 -m test_and_train.py'
                '''
            }
        }

        stage('Copy Model to Shared Folder') {
            steps {
                sh '''
                    cp weights/wildlens_multiclassifier.keras ${SHARED_MODEL_PATH}
                '''
            }
        }

        stage('Trigger API Pipeline') {
            steps {
                build job: 'update-api-model-pipeline',
                      parameters: [
                          string(name: 'MODEL_PATH', value: "${SHARED_MODEL_PATH}")
                      ],
                      wait: false
            }
        }
    }
}

}
