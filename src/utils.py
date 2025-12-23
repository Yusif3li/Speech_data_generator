import re

def clean_filename(filename):
    """Converts 'TCP/IP Protocol' -> 'TCP_IP_Protocol'"""
    # Replace symbols with underscore
    clean = re.sub(r'[^a-zA-Z0-9]', '_', filename)
    # Remove double underscores
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_')