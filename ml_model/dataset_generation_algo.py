import random
import csv
import math
from datetime import datetime, timedelta

# Settings
NUM_TASKS   = 1000
CURRENT_TIME = datetime.now()

# Function to generate a random deadline
def random_deadline():
    days   = random.randint(0, 6)
    hour   = random.randint(8, 18)
    minute = random.choice([0, 15, 30, 45])
    return CURRENT_TIME + timedelta(
        days=days,
        hours=(hour - CURRENT_TIME.hour),
        minutes=(minute - CURRENT_TIME.minute)
    )

task_names = [
    "Assignment", "Project Work", "Internship Task", "Group Study", "Hackathon",
    "Report Writing", "Online Course", "Research Paper", "Presentation", "Code Review"
]

rows = []
for _ in range(NUM_TASKS):
    name     = random.choice(task_names)
    duration = random.randint(1, 12)  # Duration in hours
    deadline = random_deadline()
    
    # Calculate time remaining until deadline in hours
    total_hours_remaining = (deadline - CURRENT_TIME).total_seconds() / 3600
    priority = max(1, math.ceil(total_hours_remaining / duration))

    days_remaining = max(1, math.ceil(total_hours_remaining / 24))
    allocated_hours = max(1, math.ceil(duration / days_remaining))

    rows.append({
        "task":            name,
        "duration":        duration,
        "deadline":        deadline.strftime("%Y-%m-%d %H:%M"),
        "priority":        priority,
        "allocated_hours": allocated_hours
    })

# Sort by priority
# rows.sort(key=lambda r: r["priority"])

# Write to csv
with open("flexible_tasks_dataset.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Generated dataset with {NUM_TASKS} rows")