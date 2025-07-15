# src/data_processing/batch_processor.py
import pandas as pd
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import logging
from .pipeline import DataPipeline

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch processing for large datasets"""
    
    def __init__(self, 
                 chunk_size: int = 10000,
                 n_workers: Optional[int] = None,
                 use_multiprocessing: bool = True):
        self.chunk_size = chunk_size
        self.n_workers = n_workers or mp.cpu_count()
        self.use_multiprocessing = use_multiprocessing
        self.logger = logging.getLogger(__name__)
    
    def process_large_file(self, 
                          file_path: str,
                          processing_function: Callable,
                          output_path: str,
                          **kwargs) -> Dict[str, Any]:
        """Process large file in chunks"""
        
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing large file: {file_path}")
        self.logger.info(f"Chunk size: {self.chunk_size}, Workers: {self.n_workers}")
        
        # Determine file type
        if file_path.suffix.lower() == '.csv':
            chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)
        elif file_path.suffix.lower() == '.parquet':
            # For parquet, we need to handle differently
            df = pd.read_parquet(file_path)
            chunk_reader = [df[i:i+self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        processed_chunks = []
        total_chunks = 0
        
        if self.use_multiprocessing:
            # Use process pool for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                
                for chunk in chunk_reader:
                    future = executor.submit(processing_function, chunk, **kwargs)
                    futures.append(future)
                    total_chunks += 1
                
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        processed_chunks.append(result)
                    except Exception as e:
                        self.logger.error(f"Chunk processing failed: {e}")
        else:
            # Sequential processing
            for chunk in chunk_reader:
                try:
                    result = processing_function(chunk, **kwargs)
                    processed_chunks.append(result)
                    total_chunks += 1
                except Exception as e:
                    self.logger.error(f"Chunk processing failed: {e}")
        
        # Combine results
        if processed_chunks:
            final_result = pd.concat(processed_chunks, ignore_index=True)
            
            # Save result
            if output_path.suffix.lower() == '.csv':
                final_result.to_csv(output_path, index=False)
            else:
                final_result.to_parquet(output_path, index=False)
            
            self.logger.info(f"Batch processing completed: {len(final_result)} records")
            
            return {
                'success': True,
                'total_chunks': total_chunks,
                'processed_chunks': len(processed_chunks),
                'output_records': len(final_result),
                'output_path': str(output_path)
            }
        else:
            self.logger.error("No chunks were processed successfully")
            return {
                'success': False,
                'total_chunks': total_chunks,
                'processed_chunks': 0,
                'error': 'No chunks processed successfully'
            }

def chunk_processing_function(chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Example chunk processing function"""
    # Apply data pipeline to chunk
    pipeline = DataPipeline()
    
    # Remove export stage for chunk processing
    pipeline.remove_stage("DataExport")
    
    result = pipeline.run(chunk, kwargs)
    
    if result['success']:
        return result['processed_data']
    else:
        raise Exception(f"Pipeline failed: {result}")

# Example usage function
def process_fraud_data_batch(input_file: str, output_file: str):
    """Process fraud detection data in batches"""
    processor = BatchProcessor(chunk_size=5000, n_workers=4)
    
    result = processor.process_large_file(
        file_path=input_file,
        processing_function=chunk_processing_function,
        output_path=output_file,
        include_aggregations=False
    )
    
    return result