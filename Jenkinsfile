pipeline {
    agent any
    
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
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                echo "Building from branch: ${env.BRANCH_NAME}"
                echo "Repository: ${env.repository_name}"
                echo "Ref: ${env.ref}"
                // Vos commandes de build ici
            }
        }
        
        stage('Test') {
            steps {
                echo 'Running tests...'
                // Vos commandes de test ici
            }
        }
    }
}
