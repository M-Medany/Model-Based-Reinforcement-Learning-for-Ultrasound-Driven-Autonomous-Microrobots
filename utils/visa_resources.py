import pyvisa

# Create a resource manager
rm = pyvisa.ResourceManager()

# List all connected resources
resources = rm.list_resources()

print(resources)