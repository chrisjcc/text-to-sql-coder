"""
Hugging Face Hub utilities for model upload.
"""
import logging
import os
from huggingface_hub import HfApi, repo_exists, create_repo
from pathlib import Path


logger = logging.getLogger(__name__)


class HubUploader:
    """
    Handles uploading models to Hugging Face Hub.
    """
    
    def __init__(self, config, hf_token: str):
        """
        Initialize hub uploader.
        
        Args:
            config: HubConfig object
            hf_token: Hugging Face API token
        """
        self.config = config
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)
    
    def get_repo_id(self) -> str:
        """Get full repository ID."""
        return f"{self.config.username}/{self.config.repo_name}"
    
    def create_model_card(self, output_dir: str):
        """
        Create README.md model card.
        
        Args:
            output_dir: Directory to save README
        """
        repo_id = self.get_repo_id()
        
        # Format model card template
        model_card = self.config.model_card_template.format(
            username=self.config.username,
            repo_name=self.config.repo_name
        )
        
        # Save to file
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        
        logger.info(f"✓ Model card created at {readme_path}")
    
    def upload_model(
        self,
        model_dir: str,
        commit_message: str = "Upload model"
    ):
        """
        Upload model to Hugging Face Hub.
        
        Args:
            model_dir: Directory containing model files
            commit_message: Git commit message
        """
        repo_id = self.get_repo_id()
        
        logger.info(f"Uploading model to {repo_id}...")
        
        # Check if repo exists
        if not repo_exists(repo_id, token=self.hf_token):
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=self.config.private,
                exist_ok=True
            )
        
        # Create model card if it doesn't exist
        readme_path = os.path.join(model_dir, "README.md")
        if not os.path.exists(readme_path):
            self.create_model_card(model_dir)
        
        # Upload all files in directory
        try:
            self.api.upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=self.hf_token,
            )
            
            logger.info(f"✓ Model uploaded to https://huggingface.co/{repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise
    
    def upload_adapter_only(
        self,
        adapter_dir: str,
        commit_message: str = "Upload LoRA adapter"
    ):
        """
        Upload only LoRA adapter files (lighter upload).
        
        Args:
            adapter_dir: Directory containing adapter
            commit_message: Commit message
        """
        logger.info("Uploading LoRA adapter only...")
        
        # Key adapter files
        adapter_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "README.md",
        ]
        
        repo_id = self.get_repo_id()
        
        # Create repo if needed
        if not repo_exists(repo_id, token=self.hf_token):
            create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=self.config.private,
                exist_ok=True
            )
        
        # Create model card
        if not os.path.exists(os.path.join(adapter_dir, "README.md")):
            self.create_model_card(adapter_dir)
        
        # Upload each file that exists
        for filename in adapter_files:
            filepath = os.path.join(adapter_dir, filename)
            if os.path.exists(filepath):
                try:
                    self.api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type="model",
                        token=self.hf_token,
                    )
                    logger.info(f"✓ Uploaded {filename}")
                except Exception as e:
                    logger.warning(f"Failed to upload {filename}: {e}")
        
        logger.info(f"✓ Adapter uploaded to https://huggingface.co/{repo_id}")


def upload_to_hub(
    model_dir: str,
    hub_config,
    hf_token: str,
    adapter_only: bool = False,
    commit_message: str = "Upload trained model"
):
    """
    Convenience function to upload model to hub.
    
    Args:
        model_dir: Directory containing model
        hub_config: HubConfig object
        hf_token: Hugging Face token
        adapter_only: Whether to upload only adapter
        commit_message: Commit message
    """
    uploader = HubUploader(hub_config, hf_token)
    
    if adapter_only:
        uploader.upload_adapter_only(model_dir, commit_message)
    else:
        uploader.upload_model(model_dir, commit_message)
