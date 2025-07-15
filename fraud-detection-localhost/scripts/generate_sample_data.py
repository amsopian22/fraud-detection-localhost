import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_sample_transactions(n_samples=1000, fraud_rate=0.05):
    """Generate synthetic transaction data"""
    
    categories = ['grocery_pos', 'gas_transport', 'misc_net', 'grocery_net', 'entertainment', 'misc_pos']
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    genders = ['M', 'F']
    jobs = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Manager', 'Analyst', 'Developer', 'Designer', 'Consultant']
    
    # Generate base data
    data = []
    for i in range(n_samples):
        # Customer info
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Chris', 'Amy', 'Robert', 'Emma']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        # Transaction timestamp
        base_time = datetime.now() - timedelta(days=30)
        transaction_time = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Generate transaction
        transaction = {
            'trans_date_trans_time': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'cc_num': f"{random.randint(1000000000000000, 9999999999999999)}",
            'merchant': f"merchant_{random.randint(1000, 9999)}",
            'category': random.choice(categories),
            'amt': round(random.uniform(5.0, 1000.0), 2),
            'first': random.choice(first_names),
            'last': random.choice(last_names),
            'gender': random.choice(genders),
            'street': f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Cedar'])} St",
            'city': f"City_{random.randint(1, 100)}",
            'state': random.choice(states),
            'zip': f"{random.randint(10000, 99999)}",
            'lat': round(random.uniform(25.0, 49.0), 4),
            'long': round(random.uniform(-125.0, -66.0), 4),
            'city_pop': random.randint(1000, 500000),
            'job': random.choice(jobs),
            'dob': f"{random.randint(1950, 2000)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'trans_num': f"{random.randint(100000000000000000000000000000000, 999999999999999999999999999999999):032x}",
            'unix_time': int(transaction_time.timestamp()),
            'merch_lat': round(random.uniform(25.0, 49.0), 4),
            'merch_long': round(random.uniform(-125.0, -66.0), 4),
            'merch_zipcode': f"{random.randint(10000, 99999)}"
        }
        
        # Determine if fraud (based on some rules)
        is_fraud = False
        if random.random() < fraud_rate:
            is_fraud = True
            # Make fraudulent transactions more suspicious
            transaction['amt'] = round(random.uniform(500.0, 2000.0), 2)  # Higher amounts
            # Distance between customer and merchant
            distance = abs(transaction['lat'] - transaction['merch_lat']) + abs(transaction['long'] - transaction['merch_long'])
            if distance < 1.0:  # Make them far apart
                transaction['merch_lat'] = round(random.uniform(25.0, 49.0), 4)
                transaction['merch_long'] = round(random.uniform(-125.0, -66.0), 4)
        
        transaction['is_fraud'] = 1 if is_fraud else 0
        data.append(transaction)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating sample fraud detection data...")
    
    # Generate datasets
    train_data = generate_sample_transactions(5000, 0.05)
    val_data = generate_sample_transactions(1000, 0.05)
    test_data = generate_sample_transactions(500, 0.05)
    
    # Create output directory
    os.makedirs('/app/data/processed', exist_ok=True)
    os.makedirs('/app/data/raw', exist_ok=True)
    
    # Save datasets
    train_data.to_csv('/app/data/raw/train_data.csv', index=False)
    val_data.to_csv('/app/data/raw/val_data.csv', index=False)
    test_data.to_csv('/app/data/raw/test_data.csv', index=False)
    
    # Save as parquet for better performance
    train_data.to_parquet('/app/data/processed/train.parquet', index=False)
    val_data.to_parquet('/app/data/processed/validation.parquet', index=False)
    test_data.to_parquet('/app/data/processed/test.parquet', index=False)
    
    print(f"Generated datasets:")
    print(f"  Training: {len(train_data)} transactions ({train_data['is_fraud'].sum()} fraud)")
    print(f"  Validation: {len(val_data)} transactions ({val_data['is_fraud'].sum()} fraud)")
    print(f"  Test: {len(test_data)} transactions ({test_data['is_fraud'].sum()} fraud)")
    print("Data saved to /app/data/")
