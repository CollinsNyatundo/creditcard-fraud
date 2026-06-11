import os
import json
import hashlib
import pytest

from app.main import validate_model_manifest

@pytest.fixture
def temp_manifest_setup():
    manifest_path = "./models/model_manifest.json"
    backup_path = "./models/model_manifest.json.bak"
    
    has_backup = False
    if os.path.exists(manifest_path):
        os.rename(manifest_path, backup_path)
        has_backup = True
        
    yield
    
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    if has_backup:
        os.rename(backup_path, manifest_path)

def test_validate_model_manifest_missing_is_ignored(temp_manifest_setup) -> None:
    # If manifest doesn't exist, it should pass without error
    validate_model_manifest()

def test_validate_model_manifest_success(temp_manifest_setup) -> None:
    # Create a temporary model asset
    asset_filename = "dummy_test_asset.pkl"
    asset_path = os.path.join("./models", asset_filename)
    
    content = b"fake model content for hashing"
    with open(asset_path, "wb") as f:
        f.write(content)
        
    expected_hash = hashlib.sha256(content).hexdigest()
    
    # Write temporary manifest
    manifest_data = {
        "artifacts": {
            asset_filename: expected_hash
        }
    }
    with open("./models/model_manifest.json", "w") as f:
        json.dump(manifest_data, f)
        
    try:
        # Should execute successfully
        validate_model_manifest()
    finally:
        if os.path.exists(asset_path):
            os.remove(asset_path)

def test_validate_model_manifest_hash_mismatch(temp_manifest_setup) -> None:
    asset_filename = "dummy_test_asset.pkl"
    asset_path = os.path.join("./models", asset_filename)
    
    content = b"fake model content for hashing"
    with open(asset_path, "wb") as f:
        f.write(content)
        
    # Write incorrect hash
    manifest_data = {
        "artifacts": {
            asset_filename: "incorrect_hash_value"
        }
    }
    with open("./models/model_manifest.json", "w") as f:
        json.dump(manifest_data, f)
        
    try:
        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            validate_model_manifest()
        assert "Manifest Cryptographic Mismatch" in str(excinfo.value)
    finally:
        if os.path.exists(asset_path):
            os.remove(asset_path)

def test_validate_model_manifest_file_not_found(temp_manifest_setup) -> None:
    # Write manifest referencing nonexistent file
    manifest_data = {
        "artifacts": {
            "nonexistent_file.pkl": "some_hash"
        }
    }
    with open("./models/model_manifest.json", "w") as f:
        json.dump(manifest_data, f)
        
    with pytest.raises(ValueError) as excinfo:
        validate_model_manifest()
    assert "not found on disk" in str(excinfo.value)
