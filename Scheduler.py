

import schedule
import time

# Define the job function
def job():
    print("Scheduled Job is running...")

# Schedule the job every 10 seconds
schedule.every(10).seconds.do(job)

# Keep the script running and check for pending jobs
while True:
    schedule.run_pending()
    time.sleep(1)