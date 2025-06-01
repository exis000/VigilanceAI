from pathlib import Path
from collections import Counter

# Set path to your YOLO label folder
label_dir = Path("data/train_final/labels")

class_counter = Counter()

for label_file in label_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            if line.strip():
                class_id = int(line.strip().split()[0])
                class_counter[class_id] += 1

# Total classes
num_classes = len(class_counter)

print(f"Total classes: {num_classes}")
print("Class distribution:")
for class_id, count in sorted(class_counter.items()):
    print(f"  Class {class_id}: {count} instances")
    
    """OUTPUT FOR THIS CODE 
    Total classes: 14
    Class distribution:    CLASS 3 , 9 AND 12 ARE HIGHEST
    Class 0: 3149 instances
    Class 1: 3381 instances
    Class 2: 2959 instances
    Class 3: 28996 instances > 9k
    Class 4: 734 instances
    Class 5: 1965 instances
    Class 6: 4280 instances
    Class 7: 2817 instances
    Class 8: 9693 instances
    Class 9: 1577 instances
    Class 10: 1432 instances
    Class 11: 1034 instances
    Class 12: 9843 instances
    Class 13: 4499 instances 53.68k
    
    
    
    """
    
    
    