import os
from pathlib import Path
from scandir import scandir

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def get_image_paths(dir_path):
    dir_path = Path (dir_path)
        
    result = []    
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append(x.path)
    return result

def get_image_unique_filestem_paths(dir_path, verbose=False):
    result = get_image_paths(dir_path)   
    result_dup = set()    
    
    for fn in result:
        bn = Path(fn).stem 
        if bn in result_dup:
            if verbose:
                result.remove(fn)
                print ("Duplicate filenames are not allowed, skipping: %s" % os.path.basename(fn))
            continue
        result_dup.add(bn)
            
    return result
    
def get_all_dir_names_startswith (dir_path, startswith):
    dir_path = Path (dir_path)
    startswith = startswith.lower()
        
    result = []    
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if x.name.lower().startswith(startswith):
                result.append ( x.name[len(startswith):] )
    return result
