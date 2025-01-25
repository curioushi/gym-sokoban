from stable_baselines3 import PPO
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint1", type=str, required=True, help="Path to first checkpoint file")
    parser.add_argument("--checkpoint2", type=str, required=True, help="Path to second checkpoint file") 
    return parser.parse_args()


def compare_parameters(model1, model2):
    """Compare parameters between two models"""
    params1 = model1.get_parameters()
    params2 = model2.get_parameters()
    
    print("\nParameter comparison:")
    for key in params1.keys():
        if key in params2:
            state_dict1 = params1[key]
            state_dict2 = params2[key]
            
            print(f"\nChecking {key}:")
            for param_name in state_dict1:
                if param_name in state_dict2:
                    param1 = state_dict1[param_name]
                    param2 = state_dict2[param_name]
                    
                    if torch.is_tensor(param1) and torch.is_tensor(param2):
                        diff = torch.abs(param1 - param2).mean().item()
                        max_diff = torch.abs(param1 - param2).max().item()
                        print(f"{param_name}:")
                        print(f"  Mean difference: {diff:.6f}")
                        print(f"  Max difference: {max_diff:.6f}")


if __name__ == "__main__":
    args = parse_args()
    
    # Load both models
    model1 = PPO.load(args.checkpoint1)
    model2 = PPO.load(args.checkpoint2)
    
    # Compare parameters
    compare_parameters(model1, model2)
