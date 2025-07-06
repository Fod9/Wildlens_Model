pipeline {
    agent {
        docker {
            image 'python:3.12'
            args '-v /home/shared/Wildlens:/home/shared/Wildlens'
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

    stages {

        stage('Check mounts') {
            steps {
            sh 'ls -l /home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks/cropped_imgs/train'
            sh 'ls -l /var/lib/jenkins/workspace/Wildlens_Training'
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

        stage('Trigger API Pipeline') {
            steps {
                build job: 'Wildlens_Backend',
                wait: false?
                parameters: [
                    string(name: 'jenkins-generic-webhook-trigger-plugin_uuid', defaultValue: '', description: '')
                ]
            }
        }
    }
}
