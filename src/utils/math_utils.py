def apply_operation_divide_conquer(tensor_list, operation):
    """
    Apply a binary operation to a list of tensors using the divide and conquer approach.

    Parameters:
    tensor_list (list): A list of tensors to which the operation will be applied.
    operation (function): A binary function that takes two tensors and returns a tensor.

    Returns:
    The result of applying the operation to all tensors in the list.
    """
    # Base case: If the list has only one tensor, return that tensor
    if len(tensor_list) == 1:
        return tensor_list[0]

    mid = len(tensor_list) // 2
    left_result = apply_operation_divide_conquer(tensor_list[:mid], operation)
    right_result = apply_operation_divide_conquer(tensor_list[mid:], operation)

    return operation(left_result, right_result)