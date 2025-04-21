def print_structure(obj, name='obj', indent=0, max_depth=4):
    """
    Print the structure of a nested object (dict, list, array, tensor, etc.)
    
    Args:
        obj: The object to print
        name: Name of the object (for printing)
        indent: Current indentation level
        max_depth: Maximum depth to traverse
    """
    indent_str = '  ' * indent
    
    # Stop if we've reached maximum recursion depth
    if indent >= max_depth:
        print(f"{indent_str}{name} = ... (max depth reached)")
        return
        
    if isinstance(obj, dict):
        print(f"{indent_str}{name} = dict with keys: {list(obj.keys())}")
        for key, value in obj.items():
            print_structure(value, f"{name}['{key}']", indent + 1, max_depth)
    
    elif isinstance(obj, list):
        print(f"{indent_str}{name} = list with length: {len(obj)}")
        if len(obj) > 0 and indent < max_depth - 1:
            print_structure(obj[0], f"{name}[0]", indent + 1, max_depth)
    
    elif isinstance(obj, tuple):
        print(f"{indent_str}{name} = tuple with length: {len(obj)}")
        if len(obj) > 0 and indent < max_depth - 1:
            print_structure(obj[0], f"{name}[0]", indent + 1, max_depth)
    
    elif 'numpy' in str(type(obj)) and hasattr(obj, 'shape'):
        print(f"{indent_str}{name} = numpy array with shape: {obj.shape}, dtype: {obj.dtype}")
    
    # Handle PyTorch tensors
    elif 'torch' in str(type(obj)):
        print(f"{indent_str}{name} = torch.Tensor with shape: {obj.shape}, dtype: {obj.dtype}, device: {obj.device}")
    
    # Handle TensorFlow tensors
    elif 'tensorflow' in str(type(obj)) or 'tf.' in str(type(obj)):
        print(f"{indent_str}{name} = tf.Tensor with shape: {obj.shape}, dtype: {obj.dtype}")
        
    elif hasattr(obj, "__dict__"):  # Custom object with attributes
        print(f"{indent_str}{name} = object of type: {type(obj).__name__}")
        if indent < max_depth - 1:
            for attr_name, attr_value in obj.__dict__.items():
                if not attr_name.startswith('_'):  # Skip private attributes
                    print_structure(attr_value, f"{name}.{attr_name}", indent + 1, max_depth)
    
    else:
        print(f"{indent_str}{name} = {type(obj).__name__}")