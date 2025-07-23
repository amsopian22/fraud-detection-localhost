#!/usr/bin/env python3
"""
Demo runner for integrated realtime service to show table output
"""

import subprocess
import sys
import time
import signal

def run_integrated_service():
    """Run the integrated service and capture output"""
    print("🚀 Starting Integrated Real-time Fraud Detection Service")
    print("   This service generates transactions, makes predictions, and displays them in table format")
    print("   It also feeds data directly to the dashboard at http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print("")
    
    try:
        # Start the integrated service
        process = subprocess.Popen(
            ["docker-compose", "exec", "ml-api", "python", "scripts/integrated_realtime_service.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping service...")
        process.terminate()
        process.wait()
        print("🏁 Service stopped!")

if __name__ == "__main__":
    run_integrated_service()