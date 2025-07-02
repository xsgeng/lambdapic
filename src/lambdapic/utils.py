import re
from .core.species import Species

def uniquify_species_names(existing_species: list[Species], new_species: list[Species]):
    """Directly modify the name attribute of species in the input list"""
    existing_names = [s.name for s in existing_species]
    
    for s in new_species:
        base_name = s.name
        pattern = re.compile(rf'^{re.escape(base_name)}(\.\d+)?$')
        max_suffix = -1
        
        # Check existing names and already processed names in current batch
        for name in existing_names + [s.name for s in new_species[:new_species.index(s)]]:
            if match := pattern.match(name):
                if suffix_part := match.group(1):
                    current_suffix = int(suffix_part[1:])
                    max_suffix = max(max_suffix, current_suffix)
                else:
                    max_suffix = max(max_suffix, 0)
        
        # Directly modify the name attribute of the original object
        if max_suffix >= 0:
            print(f"warning: species name {base_name} already exists, rename to {base_name}.{max_suffix + 1}")
            s.name = f"{base_name}.{max_suffix + 1}"
        else:
            s.name = base_name
        
        # Add the new name to the existing list
        existing_names.append(s.name)
