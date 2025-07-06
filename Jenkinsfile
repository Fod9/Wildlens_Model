pipeline {
    agent {
        node {
            label 'python-3-12'
        }
    }

    triggers {
        GenericTrigger(
            genericVariables: [
                [key: 'ref', value: '$.ref'],
                [key: 'repository_name', value: '$.repository.name'],
                [key: 'pusher_name', value: '$.pusher.name']
            ],
            token: 'wildlens_training',
            printContributedVariables: true,
            printPostContent: true,
            causeString: 'Triggered by GitHub webhook'
        )
    }

    environment {
        SHARED_MODEL_PATH = "/home/shared/Wildlens/models/multiclassifier/wildlens_multiclassifier.keras"
    }

    stages {

        stage('Debug Shared Folder') {
            steps {
                sh '''
                    echo "Checking /home/shared..."
                    ls -l /home/shared/Wildlens || echo "❌ Not accessible"

                    echo "Checking symbolic link:"
                    readlink -f data/OpenAnimalTracks || echo "❌ Broken symlink"

                    echo "Reading sample image dir:"
                    ls -l data/OpenAnimalTracks/cropped_imgs/train || echo "❌ Train folder not readable"
                '''
            }
        }

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

                        python3 -m scripts.train_and_test
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
