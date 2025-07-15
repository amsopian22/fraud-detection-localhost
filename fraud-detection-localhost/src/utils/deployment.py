import subprocess
import yaml
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .api_client import FraudDetectionClient

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Deployment management utilities"""
    
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default deployment configuration"""
        default_config = {
            'environments': {
                'localhost': {
                    'docker_compose_file': 'docker-compose.yml',
                    'env_file': '.env',
                    'health_check_url': 'http://localhost:8000/health',
                    'services': ['postgres', 'redis', 'ml-api', 'dashboard']
                },
                'staging': {
                    'docker_compose_file': 'docker-compose.staging.yml',
                    'env_file': '.env.staging',
                    'health_check_url': 'http://staging.frauddetection.com/health',
                    'services': ['postgres', 'redis', 'ml-api', 'dashboard', 'nginx']
                },
                'production': {
                    'docker_compose_file': 'docker-compose.prod.yml',
                    'env_file': '.env.production',
                    'health_check_url': 'http://api.frauddetection.com/health',
                    'services': ['postgres', 'redis', 'ml-api', 'dashboard', 'nginx', 'prometheus']
                }
            },
            'health_check': {
                'max_retries': 30,
                'retry_interval': 10,
                'timeout': 5
            },
            'backup': {
                'database_backup_path': './backups',
                'model_backup_path': './model_backups'
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def deploy(self, environment: str = 'localhost', force_rebuild: bool = False) -> bool:
        """Deploy to specified environment"""
        if environment not in self.config['environments']:
            self.logger.error(f"Unknown environment: {environment}")
            return False
        
        env_config = self.config['environments'][environment]
        
        self.logger.info(f"Starting deployment to {environment}...")
        
        try:
            # Pre-deployment checks
            if not self._pre_deployment_checks(environment):
                return False
            
            # Stop existing services
            self._stop_services(env_config)
            
            # Build images if needed
            if force_rebuild:
                self._build_images(env_config)
            
            # Start services
            if not self._start_services(env_config):
                return False
            
            # Health checks
            if not self._wait_for_health(env_config):
                return False
            
            # Post-deployment tasks
            self._post_deployment_tasks(environment)
            
            self.logger.info(f"Deployment to {environment} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False
    
    def _pre_deployment_checks(self, environment: str) -> bool:
        """Pre-deployment checks"""
        env_config = self.config['environments'][environment]
        
        # Check if Docker Compose file exists
        compose_file = Path(env_config['docker_compose_file'])
        if not compose_file.exists():
            self.logger.error(f"Docker Compose file not found: {compose_file}")
            return False
        
        # Check if environment file exists
        env_file = Path(env_config['env_file'])
        if not env_file.exists():
            self.logger.warning(f"Environment file not found: {env_file}")
        
        # Check disk space (basic check)
        if environment == 'production':
            disk_usage = subprocess.check_output(['df', '-h', '.']).decode()
            self.logger.info(f"Disk usage:\n{disk_usage}")
        
        return True
    
    def _stop_services(self, env_config: Dict[str, Any]):
        """Stop running services"""
        compose_file = env_config['docker_compose_file']
        self.logger.info("Stopping existing services...")
        
        try:
            subprocess.run([
                'docker-compose', '-f', compose_file, 'down'
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to stop services: {e.stderr.decode() if e.stderr else e.stdout.decode()}")
    
    def _build_images(self, env_config: Dict[str, Any]):
        """Build Docker images"""
        compose_file = env_config['docker_compose_file']
        self.logger.info("Building Docker images...")
        
        try:
            subprocess.run([
                'docker-compose', '-f', compose_file, 'build', '--no-cache'
            ], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to build images: {e.stderr.decode() if e.stderr else e.stdout.decode()}")
            raise
    
    def _start_services(self, env_config: Dict[str, Any]) -> bool:
        """Start services"""
        compose_file = env_config['docker_compose_file']
        env_file = env_config.get('env_file', '.env')
        
        self.logger.info("Starting services...")
        
        try:
            cmd = ['docker-compose', '-f', compose_file]
            if Path(env_file).exists():
                cmd.extend(['--env-file', env_file])
            cmd.extend(['up', '-d'])
            
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start services: {e.stderr.decode() if e.stderr else e.stdout.decode()}")
            return False
    
    def _wait_for_health(self, env_config: Dict[str, Any]) -> bool:
        """Wait for services to be healthy"""
        health_url = env_config.get('health_check_url')
        if not health_url:
            self.logger.warning("No health check URL configured")
            return True
        
        health_config = self.config['health_check']
        max_retries = health_config['max_retries']
        retry_interval = health_config['retry_interval']
        timeout = health_config['timeout']
        
        self.logger.info(f"Waiting for services to be healthy: {health_url}")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(health_url, timeout=timeout)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') in ['healthy', 'ok']:
                        self.logger.info("All services are healthy")
                        return True
                    else:
                        self.logger.info(f"Services not yet healthy: {health_data}")
                
            except Exception as e:
                self.logger.debug(f"Health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
        
        self.logger.error("Services failed to become healthy within timeout")
        return False
    
    def _post_deployment_tasks(self, environment: str):
        """Post-deployment tasks"""
        self.logger.info("Running post-deployment tasks...")
        
        try:
            # Run database migrations if needed
            if environment != 'localhost':
                self._run_database_migrations()
            
            # Warm up model cache
            self._warm_up_model(environment)
            
            # Send deployment notification
            self._send_deployment_notification(environment)
            
        except Exception as e:
            self.logger.warning(f"Post-deployment task failed: {e}")
    
    def _run_database_migrations(self):
        """Run database migrations"""
        try:
            subprocess.run([
                'docker-compose', 'exec', '-T', 'ml-api',
                'python', '-m', 'alembic', 'upgrade', 'head'
            ], check=True, capture_output=True)
            self.logger.info("Database migrations completed")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Database migration failed: {e.stderr.decode() if e.stderr else e.stdout.decode()}")
    
    def _warm_up_model(self, environment: str):
        """Warm up model cache"""
        env_config = self.config['environments'][environment]
        base_url = env_config.get('health_check_url', 'http://localhost:8000').replace('/health', '')
        
        try:
            # Make a test prediction to warm up the model
            client = FraudDetectionClient(base_url=base_url)
            test_transaction = {
                "transaction_id": "warmup-test-123",
                "cc_num": "1234567890123456",
                "merchant": "test_merchant",
                "category": "grocery_pos",
                "amt": 50.0,
                "first": "Test",
                "last": "User",
                "gender": "M",
                "street": "123 Test St",
                "city": "Test City",
                "state": "CA",
                "zip": "12345",
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 50000,
                "job": "Engineer",
                "dob": "1980-01-01",
                "trans_date_trans_time": datetime.now().isoformat(),
                "merch_lat": 40.7580,
                "merch_long": -73.9855
            }
            
            result = client.predict_single(test_transaction)
            if result:
                self.logger.info("Model warmed up successfully")
            else:
                self.logger.warning("Model warm-up failed")
                
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")
    
    def _send_deployment_notification(self, environment: str):
        """Send deployment notification"""
        # This would integrate with Slack, email, or other notification systems
        self.logger.info(f"Deployment to {environment} completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def rollback(self, environment: str) -> bool:
        """Rollback to previous version"""
        self.logger.info(f"Rolling back {environment} deployment...")
        
        try:
            # This would restore from backup or previous Docker images
            # For now, just restart services
            env_config = self.config['environments'][environment]
            self._stop_services(env_config)
            self._start_services(env_config)
            
            if self._wait_for_health(env_config):
                self.logger.info("Rollback completed successfully")
                return True
            else:
                self.logger.error("Rollback failed - services not healthy")
                return False
                
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def backup_data(self, environment: str) -> bool:
        """Backup database and models"""
        self.logger.info(f"Creating backup for {environment}...")
        
        try:
            env_config = self.config['environments'][environment]
            backup_config = self.config['backup']
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Create backup directories
            db_backup_path = Path(backup_config['database_backup_path'])
            model_backup_path = Path(backup_config['model_backup_path'])
            
            db_backup_path.mkdir(parents=True, exist_ok=True)
            model_backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup database
            db_backup_file = db_backup_path / f"backup_{environment}_{timestamp}.sql"
            docker_cmd = ['docker-compose', '-f', env_config['docker_compose_file'], 'exec', '-T', 'postgres']
            
            # Get DB credentials from env file
            env_vars = {}
            if Path(env_config['env_file']).exists():
                with open(env_config['env_file']) as f:
                    for line in f:
                        if '=' in line:
                            key, val = line.strip().split('=', 1)
                            env_vars[key] = val
            
            db_user = env_vars.get('POSTGRES_USER', 'frauduser')
            db_name = env_vars.get('POSTGRES_DB', 'frauddb')
            
            with open(db_backup_file, 'w') as f_out:
                subprocess.run(
                    docker_cmd + ['pg_dump', '-U', db_user, db_name],
                    stdout=f_out, check=True
                )
            self.logger.info(f"Database backup successful: {db_backup_file}")
            
            # Backup models
            api_container_name = self._get_container_name('ml-api', env_config)
            if api_container_name:
                model_backup_dir = model_backup_path / f"models_{environment}_{timestamp}"
                subprocess.run([
                    'docker', 'cp', f'{api_container_name}:/app/models/', str(model_backup_dir)
                ], check=True)
                self.logger.info(f"Model backup successful: {model_backup_dir}")
            else:
                self.logger.error("Could not find ml-api container to backup models.")

            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False

    def _get_container_name(self, service_name: str, env_config: Dict[str, Any]) -> Optional[str]:
        """Get the running container name for a service"""
        try:
            compose_file = env_config['docker_compose_file']
            result = subprocess.check_output(
                ['docker-compose', '-f', compose_file, 'ps', '-q', service_name],
                text=True
            ).strip()
            if result:
                return result
        except Exception as e:
            self.logger.error(f"Could not get container name for {service_name}: {e}")
        return None

# CLI interface for deployment
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fraud Detection Deployment Manager')
    parser.add_argument('action', choices=['deploy', 'rollback', 'backup', 'health'], 
                       help='Action to perform')
    parser.add_argument('--environment', '-e', default='localhost',
                       help='Target environment')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of Docker images')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    manager = DeploymentManager()
    
    if args.action == 'deploy':
        success = manager.deploy(args.environment, args.force_rebuild)
        exit(0 if success else 1)
    elif args.action == 'rollback':
        success = manager.rollback(args.environment)
        exit(0 if success else 1)
    elif args.action == 'backup':
        success = manager.backup_data(args.environment)
        exit(0 if success else 1)
    elif args.action == 'health':
        env_config = manager.config['environments'][args.environment]
        health_url = env_config.get('health_check_url')
        if health_url:
            try:
                client = FraudDetectionClient(base_url=health_url.replace('/health', ''))
                health_info = client.health_check()
                print(json.dumps(health_info, indent=2))
                if health_info.get("status") not in ["healthy", "ok"]:
                    exit(1)
            except Exception as e:
                print(f"Health check failed: {e}")
                exit(1)
        else:
            print("No health check URL configured")
            exit(1)

if __name__ == '__main__':
    main()