'''Memory prediction utilities for Lab 06.'''

def predict_training_memory(total_params: int, trainable_params: int) -> dict:
    '''Predict weight, gradient, and Adam state memory in bytes.'''

    # Calculate memory for all model weights
    weight_bytes = 4 * total_params

    # Calculate gradient memory for trainable parameters
    gradient_bytes = 4 * trainable_params

    # Calculate Adam optimizer memory for trainable parameters
    optimizer_bytes = 8 * trainable_params

    # Calculate total predicted training-state memory
    total_bytes = weight_bytes + gradient_bytes + optimizer_bytes

    return {'predicted_weight_bytes': weight_bytes, 'predicted_grad_bytes': gradient_bytes, 'predicted_optimizer_bytes': optimizer_bytes, 'predicted_total_bytes': total_bytes}