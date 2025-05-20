import logging
from typing import Dict, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from datetime import datetime

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self, connection_uri: str = "mongodb://localhost:27017/"):
        """Initialize MongoDB client with connection URI."""
        try:
            self.client = MongoClient(connection_uri)
            self.db: Database = self.client.jd_analyzer
            
            # Initialize collections - now using a single collection for JD data
            self.jd_data_collection: Collection = self.db.jd_data
            self.metadata_collection: Collection = self.db.metadata
            
            # Create indexes
            self._setup_indexes()
            
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _setup_indexes(self):
        """Set up necessary indexes for efficient querying."""
        try:
            # Indexes for jd_data collection
            self.jd_data_collection.create_index("doc_id", unique=True)
            self.jd_data_collection.create_index("core_metadata.company")
            self.jd_data_collection.create_index("core_metadata.job_title")
            self.jd_data_collection.create_index("created_at")
            self.jd_data_collection.create_index("requirements.skills.technical")
            self.jd_data_collection.create_index("requirements.skills.soft")
            
            # Metadata collection index
            self.metadata_collection.create_index("doc_id", unique=True)
            
            logger.info("Successfully created MongoDB indexes")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise

    def store_jd_data(self, doc_id: str, jd_data: Dict) -> bool:
        """Store job description data in MongoDB.
        
        Args:
            doc_id: Unique identifier for the job description
            jd_data: Extracted and processed job description data
            
        Returns:
            bool: True if storage was successful
        """
        try:
            # Prepare unified JD document
            jd_doc = {
                "doc_id": doc_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "core_metadata": jd_data.get("core_metadata", {}),
                "compensation": jd_data.get("compensation", {}),
                "requirements": jd_data.get("requirements", {}),
                "job_details": jd_data.get("job_details", {}),
                "additional_info": jd_data.get("additional_info", {})
            }
            
            # Store metadata separately
            metadata_doc = {
                "doc_id": doc_id,
                "original_filename": jd_data.get("original_filename"),
                "file_size": jd_data.get("file_size"),
                "processing_stats": jd_data.get("stats", {}),
                "created_at": datetime.utcnow()
            }
            
            # Insert documents
            self.jd_data_collection.insert_one(jd_doc)
            self.metadata_collection.insert_one(metadata_doc)
            
            logger.info(f"Successfully stored JD data for doc_id: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store JD data: {e}")
            return False

    def get_jd_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve complete job description data by doc_id.
        
        Args:
            doc_id: Unique identifier for the job description
            
        Returns:
            Dict containing complete JD data or None if not found
        """
        try:
            # Get JD data
            jd_doc = self.jd_data_collection.find_one({"doc_id": doc_id})
            if not jd_doc:
                return None
            
            # Get metadata
            metadata = self.metadata_collection.find_one(
                {"doc_id": doc_id},
                {"_id": 0, "doc_id": 0}
            )
            
            # Combine all data
            complete_data = {
                "doc_id": doc_id,
                "core_metadata": jd_doc.get("core_metadata", {}),
                "compensation": jd_doc.get("compensation", {}),
                "requirements": jd_doc.get("requirements", {}),
                "job_details": jd_doc.get("job_details", {}),
                "additional_info": jd_doc.get("additional_info", {}),
                "metadata": metadata or {}
            }
            
            return complete_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve JD data: {e}")
            return None

    def update_jd_data(self, doc_id: str, update_data: Dict) -> bool:
        """Update job description data.
        
        Args:
            doc_id: Unique identifier for the job description
            update_data: Dictionary containing updated data
            
        Returns:
            bool: True if update was successful
        """
        try:
            update_time = datetime.utcnow()
            update_fields = {}
            
            # Update fields based on what's provided
            for field in ["core_metadata", "compensation", "requirements", "job_details", "additional_info"]:
                if field in update_data:
                    update_fields[field] = update_data[field]
            
            if update_fields:
                update_fields["updated_at"] = update_time
                self.jd_data_collection.update_one(
                    {"doc_id": doc_id},
                    {"$set": update_fields}
                )
            
            logger.info(f"Successfully updated JD data for doc_id: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update JD data: {e}")
            return False

    def delete_jd(self, doc_id: str) -> bool:
        """Delete all data related to a job description.
        
        Args:
            doc_id: Unique identifier for the job description
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Delete from both collections
            self.jd_data_collection.delete_one({"doc_id": doc_id})
            self.metadata_collection.delete_one({"doc_id": doc_id})
            
            logger.info(f"Successfully deleted JD data for doc_id: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete JD data: {e}")
            return False 