# FILE: config/azure_config.py - Azure Account Configuration Manager
"""
Manages Azure account configuration per project.
Supports multiple Azure accounts (personal, work, etc.) by reading from project-specific .env file.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

class AzureConfig:
    """Azure configuration manager for NextHorizon project"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize Azure configuration from environment variables.
        
        Args:
            env_file: Path to .env file. If None, uses .env in project root.
        """
        # Load environment variables from project-specific .env file
        if env_file is None:
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
        
        if Path(env_file).exists():
            load_dotenv(env_file, override=True)
            print(f"✅ Loaded Azure config from: {env_file}")
        else:
            print(f"⚠️  No .env file found at: {env_file}")
        
        # Azure Identity (Service Principal or Managed Identity)
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        # Cosmos DB
        self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("COSMOS_KEY")
        self.cosmos_database = os.getenv("COSMOS_DATABASE", "nexthorizon")
        
        # Azure Blob Storage
        self.storage_connection_string = os.getenv("STORAGE_CONNECTION_STRING")
        self.storage_container = os.getenv("STORAGE_CONTAINER", "resumes")
        
        # Azure AD B2C (for user authentication)
        self.b2c_tenant_name = os.getenv("B2C_TENANT_NAME")
        self.b2c_client_id = os.getenv("B2C_CLIENT_ID")
        self.b2c_client_secret = os.getenv("B2C_CLIENT_SECRET")
        self.b2c_policy_name = os.getenv("B2C_POLICY_NAME", "B2C_1_signupsignin")
        self.b2c_authority = f"https://{self.b2c_tenant_name}.b2clogin.com/{self.b2c_tenant_name}.onmicrosoft.com/{self.b2c_policy_name}" if self.b2c_tenant_name else None
        
        # Account identification
        self.account_type = os.getenv("AZURE_ACCOUNT_TYPE", "personal")  # personal, akamai, etc.
        
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that required Azure credentials are configured.
        
        Returns:
            Tuple of (is_valid, list_of_missing_fields)
        """
        missing = []
        
        # Check core Azure credentials (if using Service Principal)
        if self.client_id and not self.tenant_id:
            missing.append("AZURE_TENANT_ID")
        if self.client_id and not self.subscription_id:
            missing.append("AZURE_SUBSCRIPTION_ID")
        
        # Check Cosmos DB
        if not self.cosmos_endpoint:
            missing.append("COSMOS_ENDPOINT")
        if not self.cosmos_key:
            missing.append("COSMOS_KEY")
        
        # Check Storage (optional for MVP)
        # if not self.storage_connection_string:
        #     missing.append("STORAGE_CONNECTION_STRING")
        
        # Check B2C (optional if allowing guest mode)
        # if not self.b2c_tenant_name:
        #     missing.append("B2C_TENANT_NAME")
        
        return len(missing) == 0, missing
    
    def get_credential(self):
        """
        Get Azure credential object for authentication.
        Supports Service Principal and DefaultAzureCredential.
        """
        try:
            # If Service Principal credentials are provided, use ClientSecretCredential
            if self.client_id and self.client_secret and self.tenant_id:
                from azure.identity import ClientSecretCredential
                return ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            else:
                # Otherwise, use DefaultAzureCredential (works with az login, managed identity, etc.)
                from azure.identity import DefaultAzureCredential
                return DefaultAzureCredential()
        except ImportError:
            print("⚠️  Azure Identity SDK not installed. Run: pip install azure-identity")
            return None
    
    def display_info(self):
        """Display current Azure configuration (safely, without secrets)"""
        print("\n" + "="*70)
        print("AZURE CONFIGURATION")
        print("="*70)
        print(f"Account Type:     {self.account_type}")
        print(f"Tenant ID:        {self.tenant_id[:8] + '...' if self.tenant_id else 'Not set'}")
        print(f"Subscription ID:  {self.subscription_id[:8] + '...' if self.subscription_id else 'Not set'}")
        print(f"Client ID:        {self.client_id[:8] + '...' if self.client_id else 'Not set'}")
        print(f"Client Secret:    {'***' if self.client_secret else 'Not set'}")
        print(f"\nCosmos DB:")
        print(f"  Endpoint:       {self.cosmos_endpoint if self.cosmos_endpoint else 'Not set'}")
        print(f"  Database:       {self.cosmos_database}")
        print(f"  Key:            {'***' if self.cosmos_key else 'Not set'}")
        print(f"\nBlob Storage:")
        print(f"  Connection:     {'***' if self.storage_connection_string else 'Not set'}")
        print(f"  Container:      {self.storage_container}")
        print(f"\nAzure AD B2C:")
        print(f"  Tenant:         {self.b2c_tenant_name if self.b2c_tenant_name else 'Not set'}")
        print(f"  Client ID:      {self.b2c_client_id[:8] + '...' if self.b2c_client_id else 'Not set'}")
        print(f"  Policy:         {self.b2c_policy_name}")
        print("="*70 + "\n")
        
        is_valid, missing = self.validate()
        if not is_valid:
            print(f"⚠️  Missing configuration: {', '.join(missing)}")
        else:
            print("✅ Configuration is valid!")


# Global config instance
_config_instance = None

def get_azure_config(reload: bool = False) -> AzureConfig:
    """
    Get the global Azure configuration instance.
    
    Args:
        reload: If True, reload configuration from .env file
        
    Returns:
        AzureConfig instance
    """
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = AzureConfig()
    return _config_instance


# CLI tool for testing configuration
if __name__ == "__main__":
    print("Testing Azure Configuration...")
    config = AzureConfig()
    config.display_info()
    
    is_valid, missing = config.validate()
    if is_valid:
        print("\n✅ All required Azure credentials are configured!")
        
        # Test credential creation
        print("\nTesting credential creation...")
        cred = config.get_credential()
        if cred:
            print(f"✅ Credential created: {type(cred).__name__}")
    else:
        print(f"\n❌ Configuration incomplete. Missing: {', '.join(missing)}")
        print("\nTo fix:")
        print("1. Copy .env.azure.example to .env")
        print("2. Fill in your Azure credentials")
        print("3. Run this script again")
