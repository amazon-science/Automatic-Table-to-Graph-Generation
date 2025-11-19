"""
AWS Credentials Helper
Automatically fetches credentials from EC2 instance role or environment variables
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Tuple


def get_imds_token(timeout: int = 1) -> Optional[str]:
    """Get IMDSv2 token for EC2 metadata access"""
    try:
        req = urllib.request.Request(
            'http://169.254.169.254/latest/api/token',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
            method='PUT'
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode('utf-8')
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_instance_role_name(token: str, timeout: int = 1) -> Optional[str]:
    """Get the IAM role name attached to the EC2 instance"""
    try:
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/iam/security-credentials/',
            headers={'X-aws-ec2-metadata-token': token}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            role_name = response.read().decode('utf-8').strip()
            return role_name if role_name else None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_instance_region(token: str, timeout: int = 1) -> Optional[str]:
    """Get the region of the EC2 instance"""
    try:
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/placement/region',
            headers={'X-aws-ec2-metadata-token': token}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode('utf-8').strip()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_instance_role_credentials(token: str, role_name: str, timeout: int = 1) -> Optional[Dict[str, str]]:
    """Get temporary credentials from the instance role"""
    try:
        req = urllib.request.Request(
            f'http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}',
            headers={'X-aws-ec2-metadata-token': token}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            creds_json = response.read().decode('utf-8')
            creds = json.loads(creds_json)
            
            # Validate credentials
            if creds.get('Code') == 'Success':
                # Get region from instance metadata
                region = get_instance_region(token)
                return {
                    'access_key': creds.get('AccessKeyId'),
                    'secret_key': creds.get('SecretAccessKey'),
                    'session_token': creds.get('Token'),
                    'expiration': creds.get('Expiration'),
                    'region': region,
                    'source': 'instance_role'
                }
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError):
        pass
    
    return None


def get_env_credentials() -> Optional[Dict[str, str]]:
    """Get credentials from environment variables"""
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    session_token = os.environ.get('AWS_SESSION_TOKEN')
    region = os.environ.get('AWS_DEFAULT_REGION')
    
    if access_key and secret_key:
        return {
            'access_key': access_key,
            'secret_key': secret_key,
            'session_token': session_token,
            'region': region,
            'source': 'environment'
        }
    
    return None


def get_aws_credentials(use_instance_role: bool = False) -> Optional[Dict[str, str]]:
    """
    Get AWS credentials from specified source
    
    Args:
        use_instance_role: If True, only try EC2 instance role. If False, only try environment variables.
    
    Returns:
        credentials_dict or None
        credentials_dict contains: access_key, secret_key, session_token (optional), region, source
    """
    if use_instance_role:
        # Only try instance role
        token = get_imds_token()
        if token:
            role_name = get_instance_role_name(token)
            if role_name:
                creds = get_instance_role_credentials(token, role_name)
                if creds:
                    return creds
    else:
        # Only try environment variables
        creds = get_env_credentials()
        if creds:
            return creds
    
    return None


def setup_aws_credentials(use_instance_role: bool = False) -> bool:
    """
    Setup AWS credentials in environment
    
    Args:
        use_instance_role: If True, only use EC2 instance role. If False, only use environment variables.
    
    Returns:
        bool: True if credentials are available, False otherwise
    """
    creds = get_aws_credentials(use_instance_role=use_instance_role)
    
    if creds:
        # Only set environment variables if they came from instance role
        # (if from environment, they're already set)
        if creds['source'] == 'instance_role':
            os.environ['AWS_ACCESS_KEY_ID'] = creds['access_key']
            os.environ['AWS_SECRET_ACCESS_KEY'] = creds['secret_key']
            
            if creds.get('session_token'):
                os.environ['AWS_SESSION_TOKEN'] = creds['session_token']
            
            # Set region from instance metadata if available
            if creds.get('region'):
                os.environ['AWS_DEFAULT_REGION'] = creds['region']
        
        # Set default region if still not set
        if not os.environ.get('AWS_DEFAULT_REGION'):
            os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
        
        return True
    
    return False