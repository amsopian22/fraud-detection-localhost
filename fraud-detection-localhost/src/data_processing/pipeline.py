# src/data_processing/pipeline.py
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ProcessingStageEnum(Enum):
    """Data processing stages"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    EXPORT = "export"

@dataclass
class ProcessingResult:
    """Result of a processing stage"""
    stage: ProcessingStageEnum
    success: bool
    message: str
    data_shape: tuple
    processing_time: float
    metadata: Dict[str, Any]

class ProcessingStage(ABC):
    """Abstract base class for processing stages"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Process the data"""
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        return data is not None and not data.empty

class DataIngestionStage(ProcessingStage):
    """Data ingestion and loading stage"""
    
    def __init__(self):
        super().__init__("DataIngestion")
        self.supported_formats = ['.csv', '.parquet', '.json']
    
    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Load data from various sources"""
        source_path = context.get('source_path')
        if source_path:
            return self.load_from_file(source_path)
        
        # If data is already provided, return as-is
        return data
    
    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        self.logger.info(f"Loaded data from {file_path}: {df.shape}")
        return df
    
    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from database"""
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        self.logger.info(f"Loaded data from database: {df.shape}")
        return df

class DataValidationStage(ProcessingStage):
    """Data validation and quality checks"""
    
    def __init__(self):
        super().__init__("DataValidation")
        self.required_columns = [
            'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
            'first', 'last', 'gender', 'lat', 'long', 'city_pop', 'is_fraud'
        ]
    
    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Validate data quality and structure"""
        self.logger.info("Starting data validation...")
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        self._validate_data_types(data)
        
        # Check for excessive missing values
        self._validate_missing_values(data)
        
        # Check value ranges
        self._validate_value_ranges(data)
        
        self.logger.info("Data validation completed successfully")
        return data
    
    def _validate_data_types(self, data: pd.DataFrame):
        """Validate data types"""
        # Amount should be numeric
        if not pd.api.types.is_numeric_dtype(data['amt']):
            data['amt'] = pd.to_numeric(data['amt'], errors='coerce')
        
        # Coordinates should be numeric
        for coord_col in ['lat', 'long', 'merch_lat', 'merch_long']:
            if coord_col in data.columns and not pd.api.types.is_numeric_dtype(data[coord_col]):
                data[coord_col] = pd.to_numeric(data[coord_col], errors='coerce')
        
        # City population should be numeric
        if 'city_pop' in data.columns and not pd.api.types.is_numeric_dtype(data['city_pop']):
            data['city_pop'] = pd.to_numeric(data['city_pop'], errors='coerce')
    
    def _validate_missing_values(self, data: pd.DataFrame):
        """Check for excessive missing values"""
        missing_threshold = 0.5  # 50% threshold
        
        for column in self.required_columns:
            if column in data.columns:
                missing_rate = data[column].isnull().mean()
                if missing_rate > missing_threshold:
                    raise ValueError(f"Column {column} has {missing_rate:.2%} missing values")
    
    def _validate_value_ranges(self, data: pd.DataFrame):
        """Validate value ranges"""
        # Amount should be positive
        if (data['amt'] <= 0).any():
            self.logger.warning("Found non-positive amounts")
        
        # Coordinates should be within valid ranges
        if 'lat' in data.columns:
            invalid_lat = (data['lat'] < -90) | (data['lat'] > 90)
            if invalid_lat.any():
                self.logger.warning(f"Found {invalid_lat.sum()} invalid latitude values")
        
        if 'long' in data.columns:
            invalid_long = (data['long'] < -180) | (data['long'] > 180)
            if invalid_long.any():
                self.logger.warning(f"Found {invalid_long.sum()} invalid longitude values")

class DataCleaningStage(ProcessingStage):
    """Data cleaning and preprocessing"""
    
    def __init__(self):
        super().__init__("DataCleaning")
    
    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Clean and preprocess data"""
        self.logger.info("Starting data cleaning...")
        
        original_shape = data.shape
        
        # Remove duplicates
        data = self._remove_duplicates(data)
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Fix data quality issues
        data = self._fix_data_quality_issues(data)
        
        # Remove outliers
        data = self._handle_outliers(data)
        
        final_shape = data.shape
        self.logger.info(f"Data cleaning completed: {original_shape} -> {final_shape}")
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_count = len(data)
        
        # Remove exact duplicates
        data = data.drop_duplicates()
        
        # Remove duplicates based on transaction number if available
        if 'trans_num' in data.columns:
            data = data.drop_duplicates(subset=['trans_num'])
        
        duplicates_removed = initial_count - len(data)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate records")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Fill missing amounts with median
        if 'amt' in data.columns and data['amt'].isnull().any():
            median_amt = data['amt'].median()
            data['amt'].fillna(median_amt, inplace=True)
            self.logger.info(f"Filled missing amounts with median: {median_amt}")
        
        # Fill missing coordinates with mode or drop
        coordinate_columns = ['lat', 'long', 'merch_lat', 'merch_long']
        for col in coordinate_columns:
            if col in data.columns and data[col].isnull().any():
                missing_count = data[col].isnull().sum()
                if missing_count < len(data) * 0.1:  # Less than 10% missing
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    self.logger.warning(f"High missing rate for {col}: {missing_count}/{len(data)}")
        
        # Fill missing categorical values
        categorical_columns = ['gender', 'category', 'state']
        for col in categorical_columns:
            if col in data.columns and data[col].isnull().any():
                mode_value = data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown'
                data[col].fillna(mode_value, inplace=True)
        
        return data
    
    def _fix_data_quality_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix common data quality issues"""
        # Fix negative amounts
        if 'amt' in data.columns:
            negative_amounts = data['amt'] < 0
            if negative_amounts.any():
                data.loc[negative_amounts, 'amt'] = data.loc[negative_amounts, 'amt'].abs()
                self.logger.info(f"Fixed {negative_amounts.sum()} negative amounts")
        
        # Fix invalid coordinates
        if 'lat' in data.columns:
            invalid_lat = (data['lat'] < -90) | (data['lat'] > 90)
            if invalid_lat.any():
                data = data[~invalid_lat]
                self.logger.info(f"Removed {invalid_lat.sum()} records with invalid latitude")
        
        if 'long' in data.columns:
            invalid_long = (data['long'] < -180) | (data['long'] > 180)
            if invalid_long.any():
                data = data[~invalid_long]
                self.logger.info(f"Removed {invalid_long.sum()} records with invalid longitude")
        
        # Standardize text fields
        text_columns = ['gender', 'category', 'state']
        for col in text_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip().str.upper()
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns"""
        # Handle amount outliers
        if 'amt' in data.columns:
            Q1 = data['amt'].quantile(0.25)
            Q3 = data['amt'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 3 * IQR  # More lenient upper bound for amounts
            
            outliers = (data['amt'] < lower_bound) | (data['amt'] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Cap outliers instead of removing them
                data.loc[data['amt'] < lower_bound, 'amt'] = lower_bound
                data.loc[data['amt'] > upper_bound, 'amt'] = upper_bound
                self.logger.info(f"Capped {outlier_count} amount outliers")
        
        return data

class FeatureEngineeringStage(ProcessingStage):
    """Feature engineering stage"""
    
    def __init__(self):
        super().__init__("FeatureEngineering")
    
    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature engineering"""
        self.logger.info("Starting feature engineering...")
        
        from feature_engineering import MasterFeatureEngineer
        
        engineer = MasterFeatureEngineer()
        include_aggregations = context.get('include_aggregations', False)
        
        engineered_data = engineer.create_all_features(data, include_aggregations=include_aggregations)
        
        self.logger.info(f"Feature engineering completed: {data.shape} -> {engineered_data.shape}")
        return engineered_data

class DataExportStage(ProcessingStage):
    """Data export stage"""
    
    def __init__(self):
        super().__init__("DataExport")
    
    def process(self, data: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Export processed data"""
        output_path = context.get('output_path')
        if not output_path:
            self.logger.warning("No output path specified, skipping export")
            return data
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        if output_path.suffix.lower() == '.csv':
            data.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.parquet':
            data.to_parquet(output_path, index=False)
        else:
            # Default to parquet
            output_path = output_path.with_suffix('.parquet')
            data.to_parquet(output_path, index=False)
        
        self.logger.info(f"Exported data to: {output_path}")
        return data

class DataPipeline:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.stages = []
        self.logger = logging.getLogger(__name__)
        
        # Default pipeline stages
        self.add_stage(DataIngestionStage())
        self.add_stage(DataValidationStage())
        self.add_stage(DataCleaningStage())
        self.add_stage(FeatureEngineeringStage())
        self.add_stage(DataExportStage())
    
    def add_stage(self, stage: ProcessingStage):
        """Add a processing stage to the pipeline"""
        self.stages.append(stage)
    
    def remove_stage(self, stage_name: str):
        """Remove a processing stage by name"""
        self.stages = [stage for stage in self.stages if stage.name != stage_name]
    
    def run(self, data: pd.DataFrame = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        if context is None:
            context = {}
        
        self.logger.info("Starting data pipeline...")
        start_time = datetime.now()
        
        results = []
        current_data = data
        
        for i, stage in enumerate(self.stages):
            stage_start_time = datetime.now()
            
            try:
                self.logger.info(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")
                
                # Validate input
                if current_data is not None and not stage.validate_input(current_data):
                    raise ValueError(f"Invalid input for stage {stage.name}")
                
                # Process data
                processed_data = stage.process(current_data, context)
                
                # Record result
                stage_time = (datetime.now() - stage_start_time).total_seconds()
                result = ProcessingResult(
                    stage=ProcessingStageEnum(stage.name.lower()),
                    success=True,
                    message="Completed successfully",
                    data_shape=processed_data.shape if processed_data is not None else (0, 0),
                    processing_time=stage_time,
                    metadata={}
                )
                
                results.append(result)
                current_data = processed_data
                
                self.logger.info(f"Stage {stage.name} completed in {stage_time:.2f}s")
                
            except Exception as e:
                stage_time = (datetime.now() - stage_start_time).total_seconds()
                result = ProcessingResult(
                    stage=ProcessingStageEnum(stage.name.lower()),
                    success=False,
                    message=str(e),
                    data_shape=(0, 0),
                    processing_time=stage_time,
                    metadata={}
                )
                
                results.append(result)
                self.logger.error(f"Stage {stage.name} failed: {e}")
                
                # Stop pipeline on failure
                break
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Compile final results
        pipeline_result = {
            'success': all(result.success for result in results),
            'total_time': total_time,
            'stages': results,
            'final_data_shape': current_data.shape if current_data is not None else (0, 0),
            'processed_data': current_data
        }
        
        self.logger.info(f"Pipeline completed in {total_time:.2f}s")
        return pipeline_result