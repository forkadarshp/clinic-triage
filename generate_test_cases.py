#!/usr/bin/env python3
"""Generate 164 diverse test cases for evaluation."""

import json
import random
from pathlib import Path
from src import config

# Seed for reproducibility
random.seed(42)

AGES = [8, 12, 16, 18, 22, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
GENDERS = ["male", "female"]
MINUTES = [10, 15, 20, 25, 30, 45, 60]

LOCATIONS = [
    "123 Oak Street",
    "456 Elm Ave",
    "789 Pine Street",
    "22 Market St",
    "5th Ave subway entrance",
    "Highway 101 rest stop",
    "Downtown Mall food court, table 12",
    "City Park gazebo",
    "Green Hills Assisted Living dining room",
    "Riverside Apartments lobby",
    "Union Station platform",
    "Lakeside Trailhead",
    "Maple Clinic parking lot",
    "Central Library entrance",
    "Harborview Pier",
    "Northside Gym",
    "Westside Grocery",
    "Hilltop Church foyer",
    "Pinecrest School front office",
    "Oakridge Pharmacy",
    "Maplewood Community Center",
    "Eastside Bus Terminal",
    "Willow Creek Trail parking",
    "Summit High School gym",
]

DEPARTMENTS = [
    "Pediatrics", "Orthopedics", "OB/GYN", "Urgent Care", "Urology",
    "Infectious Disease", "ENT", "Ophthalmology", "Emergency Medicine",
    "Cardiology", "Neurology", "General Surgery"
]

SPECIALTIES = [
    "Endocrinology", "Internal Medicine", "Family Medicine", "Sports Medicine",
    "Gastroenterology", "Pulmonology", "Cardiology", "Dermatology",
    "Physical Therapy", "Psychiatry", "Rheumatology", "Nephrology"
]

def generate_emergency(idx: int) -> dict:
    """Generate emergency case."""
    age = random.choice(AGES)
    gender = random.choice(GENDERS)
    mins = random.choice(MINUTES)
    loc = random.choice(LOCATIONS)
    
    templates = [
        f"{age}yo {gender}, crushing chest pain radiating to left arm for {mins} minutes. Diaphoretic, pale, BP 90/60. History of MI. Currently at {loc}.",
        f"{age}yo {gender}, sudden onset slurred speech, facial drooping, right arm weakness. Symptoms started {mins} minutes ago. Address: {loc}.",
        f"{age}yo {gender} found unresponsive and not breathing, bystander started CPR. Location: {loc}.",
        f"{age}yo {gender} with anaphylaxis after eating peanuts, wheezing, hives, hypotensive. Location: {loc}.",
        f"{age}yo {gender} with severe trauma after fall from ladder, uncontrolled bleeding. Location: {loc}.",
        f"{age}yo {gender} with severe asthma attack, cannot speak full sentences, cyanotic. Location: {loc}.",
        f"{age}yo {gender} with sudden severe headache and vomiting, worst of life. Location: {loc}.",
        f"{age}yo {gender} in motorcycle accident, unresponsive, profuse bleeding. Bystander calling from {loc}.",
        f"{age}yo {gender} with cardiac arrest, no pulse detected, CPR in progress. Location: {loc}.",
        f"{age}yo {gender} with severe respiratory distress, oxygen saturation 85%, struggling to breathe. Location: {loc}.",
    ]
    
    query = templates[idx % len(templates)]
    return {
        "query": query,
        "tool": "trigger_emergency_response",
        "arguments": {"location": loc, "severity": "CRITICAL"}
    }

def generate_urgent(idx: int) -> dict:
    """Generate urgent consult case."""
    age = random.choice(AGES)
    gender = random.choice(GENDERS)
    
    templates = [
        (f"{age}yo {gender} with 104.5°F fever for 2 days, refuses to eat, lethargic. No rash. Parents concerned.",
         "Pediatrics", ["high fever", "lethargy", "decreased appetite"]),
        (f"{age}yo {gender}, fell off ladder, visible deformity right forearm, severe pain, unable to move wrist.",
         "Orthopedics", ["arm deformity", "severe pain", "limited mobility"]),
        (f"{age}yo {gender}, severe lower abdominal pain for 6 hours, nausea, last period was 8 weeks ago.",
         "OB/GYN", ["severe abdominal pain", "nausea", "missed period"]),
        (f"{age}yo {gender} with deep laceration to hand from kitchen knife, bleeding controlled, needs evaluation.",
         "Urgent Care", ["deep laceration", "bleeding", "hand pain"]),
        (f"{age}yo {gender} with flank pain radiating to groin, vomiting, suspected kidney stone.",
         "Urology", ["flank pain", "vomiting", "severe pain"]),
        (f"{age}yo {gender} with red, swollen leg and fever, cellulitis spreading.",
         "Infectious Disease", ["leg swelling", "fever", "skin infection"]),
        (f"{age}yo {gender} with severe sore throat, muffled voice, difficulty swallowing.",
         "ENT", ["severe sore throat", "difficulty swallowing", "muffled voice"]),
        (f"{age}yo {gender} with sudden vision loss in one eye, stable vitals.",
         "Ophthalmology", ["vision loss", "eye pain", "blurred vision"]),
        (f"{age}yo {gender} with suspected fracture after fall, unable to bear weight, severe swelling.",
         "Orthopedics", ["fracture", "severe pain", "swelling", "inability to bear weight"]),
        (f"{age}yo {gender} with severe ear pain and discharge, hearing loss in affected ear.",
         "ENT", ["severe ear pain", "discharge", "hearing loss"]),
    ]
    
    query, dept, symptoms = templates[idx % len(templates)]
    return {
        "query": query,
        "tool": "schedule_urgent_consult",
        "arguments": {"department": dept, "symptoms": symptoms}
    }

def generate_routine(idx: int) -> dict:
    """Generate routine care case."""
    age = random.choice(AGES)
    gender = random.choice(GENDERS)
    
    templates = [
        (f"{age}yo {gender} needs quarterly HbA1c check and insulin refill. Blood sugars stable on current regimen.",
         "lab work and prescription refill", "Endocrinology"),
        (f"{age}yo {gender} requesting annual physical. No complaints. Last checkup was 14 months ago.",
         "annual physical", "Internal Medicine"),
        (f"{age}yo {gender} with mild headache and runny nose for 2 days. Afebrile. Work stress. Sleeping 4 hours/night.",
         "sick visit", "Family Medicine"),
        (f"{age}yo {gender} with recurring knee pain after basketball. No acute injury, pain worse with stairs. Wants sports medicine eval.",
         "sports injury consultation", "Sports Medicine"),
        (f"{age}yo {gender} needs medication refill for hypertension, stable readings.",
         "prescription refill", "Internal Medicine"),
        (f"{age}yo {gender} scheduling colonoscopy screening at age 50, no symptoms.",
         "screening visit", "Gastroenterology"),
        (f"{age}yo {gender} follow-up for stable asthma, needs routine check.",
         "follow-up visit", "Pulmonology"),
        (f"{age}yo {gender} requesting vaccination update and wellness visit.",
         "vaccination visit", "Family Medicine"),
        (f"{age}yo {gender} needs routine diabetic eye exam referral.",
         "routine eye exam", "Ophthalmology"),
        (f"{age}yo {gender} requesting birth control prescription renewal.",
         "prescription refill", "OB/GYN"),
    ]
    
    query, visit_type, specialty = templates[idx % len(templates)]
    return {
        "query": query,
        "tool": "routine_care_referral",
        "arguments": {"type": visit_type, "specialty": specialty}
    }

def main():
    """Generate 164 test cases: 55 emergency, 55 urgent, 54 routine."""
    test_cases = []
    
    # Generate 55 emergency cases
    for i in range(55):
        test_cases.append(generate_emergency(i))
    
    # Generate 55 urgent cases
    for i in range(55):
        test_cases.append(generate_urgent(i))
    
    # Generate 54 routine cases
    for i in range(54):
        test_cases.append(generate_routine(i))
    
    # Shuffle for diversity
    random.shuffle(test_cases)
    
    # Write to file
    output_path = config.TEST_DATA_PATH
    with open(output_path, "w") as f:
        for case in test_cases:
            f.write(json.dumps(case) + "\n")
    
    print(f"✅ Generated {len(test_cases)} test cases")
    print(f"✅ Saved to {output_path}")
    
    # Print breakdown
    by_tool = {}
    for case in test_cases:
        tool = case["tool"]
        by_tool[tool] = by_tool.get(tool, 0) + 1
    
    print("\n📊 Breakdown by tool:")
    for tool, count in sorted(by_tool.items()):
        print(f"  {tool}: {count}")

if __name__ == "__main__":
    main()
