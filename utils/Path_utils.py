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
    
    result_len = len(result)
    result_dup = result[:]
    
    for i in range(0, result_len):
        i_path = Path(result_dup[i])
        for j in range(i+1, result_len):
            j_path = Path(result_dup[j])            
            if ( i_path.stem == j_path.stem ):                
                if verbose:
                    print ("Duplicate filenames are not allowed, skipping: %s" % (j_path.name) )                    
                result.remove (result_dup[j])                
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
