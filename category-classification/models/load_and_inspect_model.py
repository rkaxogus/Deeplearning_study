import os
import torch

def load_and_inspect_model(pth_file_path, use_gpu=True):
    """
    Load and inspect a PyTorch model's state_dict with optional GPU usage.

    Args:
        pth_file_path (str): Path to the .pth file to inspect.
        use_gpu (bool): Whether to use GPU for loading the model.

    Returns:
        None
    """
    if not os.path.exists(pth_file_path):
        print(f"Error: The file '{pth_file_path}' does not exist.")
        return

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    try:
        # Load the state_dict
        state_dict = torch.load(pth_file_path, map_location=device)

        # Display the keys in the state_dict
        print("\nModel state_dict keys:")
        for key in state_dict.keys():
            print(f"  {key}")
    
        # Display the number of parameters in the model
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\nTotal number of parameters: {total_params}")

        # Check if additional metrics are present
        if isinstance(state_dict, dict):
            for k in ["loss", "accuracy"]:
                if k in state_dict:
                    print(f"{k.capitalize()}: {state_dict[k]}")

    except Exception as e:
        print(f"Error loading the model: {e}")

def save_model_with_metrics(model, optimizer, loss, accuracy, save_path):
    """
    Save the model state_dict along with training metrics.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        optimizer (torch.optim.Optimizer): Optimizer used in training.
        loss (float): Final loss value.
        accuracy (float): Final accuracy value.
        save_path (str): Path to save the .pth file.

    Returns:
        None
    """
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }, save_path)
        print(f"Model and metrics saved to {save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")

if __name__ == "__main__":
    # Example usage
    model_path = os.path.join("models", "camping_model.pth")
    load_and_inspect_model(model_path, use_gpu=True)