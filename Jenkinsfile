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

        stage('Precheck: Shared Volume & Dataset') {
            steps {
                sh '''
                    echo "=== Precheck: Debug Info ==="
                    echo "Hostname: $(hostname)"
                    echo "User: $(whoami)"
                    echo "CWD: $(pwd)"
                    echo ""

                    echo "=== Mount Check ==="
                    mount | grep /home/shared && echo "/home/shared is mounted" || echo "/home/shared is NOT mounted"
                    echo ""

                    echo "=== Autofs Trigger ==="
                    # Try to trigger autofs or confirm dataset exists
                    ls -ld /home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks || echo "Autofs trigger failed or path missing"
                    echo ""

                    echo "=== Final Check: Train Directory ==="
                    ls -l /home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks/cropped_imgs/train || {
                        echo "ERROR: Dataset path is not accessible.";
                        exit 1;
                    }
                '''
            }
        }

        stage('Check Symlink') {
            steps {
                sh '''
                    ls -l /home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks/cropped_imgs/train
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
