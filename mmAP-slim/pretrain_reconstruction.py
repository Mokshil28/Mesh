import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multimae.criterion import MaskedL1Loss
from run_pretraining_multimae import get_model, get_args
from run_pretraining_heatmap import get_args, get_model, MaskedL1Loss
import random
import glob
import os

def load_numpy(file_path):
    input_data_np = np.load(file_path)
    
    if input_data_np.ndim == 2:
        input_data_np = input_data_np[np.newaxis, np.newaxis, :, :]
    elif input_data_np.ndim == 3:
        input_data_np = input_data_np[np.newaxis, :, :, :]
    
    input_data_tensor = torch.from_numpy(input_data_np).float()
    input_data_tensor = 2 * (input_data_tensor - input_data_tensor.min()) / (input_data_tensor.max() - input_data_tensor.min()) - 1
        
    return input_data_tensor

def save_individual_image(data, output_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='viridis')
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def display_numpy(original, masked, reconstructed, output_path, domain, filename, checkpoint_name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    original_np = original.squeeze().cpu().numpy()
    masked_np = masked.squeeze().cpu().numpy()
    reconstructed_np = reconstructed.squeeze().detach().cpu().numpy()
    
    vmin = min(original_np.min(), masked_np.min(), reconstructed_np.min())
    vmax = max(original_np.max(), masked_np.max(), reconstructed_np.max())
    
    im1 = ax1.imshow(original_np, vmin=vmin, vmax=vmax)
    ax1.set_title(f'Original {domain}\nMin: {original_np.min():.4f}, Max: {original_np.max():.4f}')
    ax1.axis('off')
    
    im2 = ax2.imshow(masked_np, vmin=vmin, vmax=vmax)
    ax2.set_title(f'Masked {domain}\nMin: {masked_np.min():.4f}, Max: {masked_np.max():.4f}')
    ax2.axis('off')
    
    im3 = ax3.imshow(reconstructed_np, vmin=vmin, vmax=vmax)
    ax3.set_title(f'Reconstructed {domain}\nMin: {reconstructed_np.min():.4f}, Max: {reconstructed_np.max():.4f}')
    ax3.axis('off')
    
    l1_loss = np.mean(np.abs(original_np - reconstructed_np))
    plt.suptitle(f'File: {filename}\nCheckpoint: {checkpoint_name}\nL1 Loss: {l1_loss:.4f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual images
    save_individual_image(original_np, output_path.parent / f'{domain}_original.png')
    save_individual_image(masked_np, output_path.parent / f'{domain}_masked.png')
    save_individual_image(reconstructed_np, output_path.parent / f'{domain}_reconstructed.png')



def process_domain(domain, output_dir):
    args = get_args()
    device = torch.device(args.device)
    
    yaml_file_path = 'cfgs/pretrain/sim_aug.yaml'
    with open(yaml_file_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    for key, value in yaml_config.items():
        setattr(args, key, value)

    args.in_domains = [domain]
    args.out_domains = [domain]

    model = get_model(args)
    model.to(device)
    
    checkpoint_path = 'output/pretrain/sim-aug_FINAL/checkpoint-39.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_name = Path(checkpoint_path).name
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    domain_codes = {'angle': 'ta', 'doppler': 'td', 'range': 'tr'}
    domain_code = domain_codes[domain]
        
    file_pattern = f'/data/sxu7/heatmaps/recorded/finetune-cls128/eval/{domain}/*/*_{domain_code}*.npy'
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        raise ValueError(f"No files found matching the pattern: {file_pattern}")
    
    numpy_path = random.choice(all_files)
    filename = Path(numpy_path).name
    print(f"Randomly selected file for {domain}: {filename}")

    input_data = load_numpy(numpy_path).to(device)
    
    with torch.no_grad():
        preds, masks = model(
            {domain: input_data},
            num_encoded_tokens=args.num_encoded_tokens,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
            fp32_output_adapters=args.fp32_output_adapters.split('-')
        )
    
    reconstructed_data = preds[domain]
    masked_data = input_data - masks[domain]  # This is the correct masked data

    loss_fn = MaskedL1Loss()
    loss = loss_fn(reconstructed_data, input_data)
    print(f"Reconstruction loss for {domain}: {loss.item()}")

    print(f"Original data - Min: {input_data.min().item():.4f}, Max: {input_data.max().item():.4f}")
    print(f"Masked data - Min: {masked_data.min().item():.4f}, Max: {masked_data.max().item():.4f}")
    print(f"Reconstructed data - Min: {reconstructed_data.min().item():.4f}, Max: {reconstructed_data.max().item():.4f}")

    output_path = output_dir / f'pretrain_reconstruction_{domain}.png'
    display_numpy(input_data, masked_data, reconstructed_data, output_path, domain, filename, checkpoint_name)
    print(f"Output images saved in {output_dir}")


def main():
    output_dir = Path('reconstruction_output')
    output_dir.mkdir(exist_ok=True)
    
    for domain in ['angle', 'doppler', 'range']:
        print(f"\nProcessing {domain} domain:")
        process_domain(domain, output_dir)

if __name__ == '__main__':
    main()
