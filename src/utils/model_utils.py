import torch

def calculate_class_weights(labels, normalize=True):
    """
    Calculate class weights inversely proportional to class frequencies.

    Args:
        labels (torch.Tensor): Tensor containing class labels.
        normalize (bool): Whether to normalize the class weights. Default is True.

    Returns:
        torch.Tensor: Tensor containing class weights.
    """
    class_weights = torch.zeros(len(torch.unique(labels)), dtype=torch.float)
    
    # Calculate class frequencies
    class_counts = torch.bincount(labels)
    
    # Calculate total number of samples
    total_samples = torch.sum(class_counts).item()
    
    # Calculate class weights inversely proportional to class frequencies
    for i in range(len(class_counts)):
        class_weights[i] = total_samples / (len(class_counts) * class_counts[i].item())
    
    # Normalize
    if normalize:
        class_weights /= torch.sum(class_weights)
    
    return class_weights