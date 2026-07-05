import random
import re


def parse_attributes(filename):
    """
    Parse attributes.txt file and return a dictionary of attribute categories and their values.
    """
    attributes = {}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by double newlines to separate categories
    sections = content.strip().split('\n\n')
    
    for section in sections:
        lines = section.strip().split('\n')
        for line in lines:
            if ':' in line:
                category, values = line.split(':', 1)
                category = category.strip()
                # Split by comma and clean up values
                value_list = [v.strip() for v in values.split(',') if v.strip()]
                attributes[category] = value_list
    
    return attributes


def parse_list_file(filename):
    """
    Parse a simple list file (subject.txt or interactions.txt) and return a list of items.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Filter out empty lines and strip whitespace
    items = [line.strip() for line in lines if line.strip()]
    
    return items


def sample_attributes(attributes_dict, num_samples=4):
    """
    Sample attributes from the attributes dictionary.
    Returns a list of tuples (category, value).
    """
    sampled = []
    
    # Get all categories
    categories = list(attributes_dict.keys())
    
    # Randomly select categories (without replacement)
    selected_categories = random.sample(categories, min(num_samples, len(categories)))
    
    for category in selected_categories:
        # Randomly select one value from the category
        value = random.choice(attributes_dict[category])
        sampled.append((category, value))
    
    return sampled


def generate_sample():
    """
    Generate a complete sample with one subject, 4 attributes (one per person), and 2 interactions.
    Returns one subject that applies to all 4 people, with each person having a different attribute.
    """
    # Parse the files
    attributes = parse_attributes('attributes.txt')
    subjects = parse_list_file('subject.txt')
    interactions = parse_list_file('interactions.txt')
    
    # Sample one subject (this will be the job/profession for all 4 people)
    subject = random.choice(subjects)
    
    # Sample 4 attributes (each will be assigned to a different person)
    sampled_attributes = sample_attributes(attributes, num_samples=4)
    
    # Sample 2 interactions (interaction 1 for persons 1&2, interaction 2 for persons 3&4)
    sampled_interactions = random.sample(interactions, min(2, len(interactions)))
    
    # Create result dictionary
    result = {
        'subject': subject,  # One job for all 4 people
        'attributes': sampled_attributes,  # 4 different attributes, one per person
        'interactions': sampled_interactions  # 2 interactions
    }
    
    return result


def print_sample(sample):
    """
    Pretty print a sample.
    """
    print("=" * 50)
    print(f"Subject: {sample['subject']}")
    print("\nAttributes:")
    for category, value in sample['attributes']:
        print(f"  - {category}: {value}")
    print("\nInteractions:")
    for interaction in sample['interactions']:
        print(f"  - {interaction}")
    print("=" * 50)


def generate_multiple_samples(num_samples=5):
    """
    Generate multiple samples.
    """
    samples = []
    for i in range(num_samples):
        print(f"\nSample {i + 1}:")
        sample = generate_sample()
        print_sample(sample)
        samples.append(sample)
    
    return samples


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    
    # Generate a single sample
    print("Generating a single sample:\n")
    sample = generate_sample()
    print_sample(sample)
    
    # Generate multiple samples
    print("\n\nGenerating 3 more samples:\n")
    generate_multiple_samples(3)
