import os
import argparse
from trainer import Trainer
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    
    # Trainer for training and inference.
    trainer = Trainer(config) 

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'inference':
        trainer.inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--submodel', type=str, default="CrossLoss", help="Chose model subtype: Prot, CrossLoss, Ligand, RL, NoTarget", choices=['Prot', 'CrossLoss', 'Ligand', 'RL', 'NoTarget'])

    # Model configuration.
    parser.add_argument('--act', type=str, default="relu", help="Activation function for the model.", choices=['relu', 'tanh', 'leaky', 'sigmoid'])
    
    parser.add_argument('--z_dim', type=int, default=16, help='Prior noise for the first GAN')
    
    parser.add_argument('--max_atom', type=int, default=45, help='Max atom number for molecules must be specified.')    
    
    parser.add_argument('--lambda_gp', type=float, default=1, help='Gradient penalty lambda multiplier for the first GAN.')
    
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the Transformer Encoder model for GAN1.')
    
    parser.add_argument('--depth', type=int, default=1, help='Depth of the Transformer model from the first GAN.')
    
    parser.add_argument('--heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the first GAN.')
    
    parser.add_argument('--dec_depth', type=int, default=1, help='Depth of the Transformer model from the second GAN.')
    
    parser.add_argument('--dec_heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the second GAN.')   
    
    parser.add_argument('--dec_dim', type=int, default=128, help='Dimension of the Transformer Decoder model for GAN2.')
     
    parser.add_argument('--mlp_ratio', type=int, default=3, help='MLP ratio for the Transformers.')
    
    parser.add_argument('--warm_up_steps', type=float, default=0, help=' Warm up steps for the first GAN.')
    
    parser.add_argument('--dis_select', type=str, default="mlp", help="Select the discriminator for the first and second GAN.")
    
    parser.add_argument('--init_type', type=str, default="normal", help="Initialization type for the model.")
    
    """parser.add_argument('--g_conv_dim',default=[128, 256, 512, 1024], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--la', type=float, default=0.5, help="lambda value for Total Discriminator loss balance")
    parser.add_argument('--la2', type=float, default=0.5, help="lambda value for Total Generator loss balance") 
    parser.add_argument('--gcn_depth', type=int, default=0, help="GCN layer depth")""" 

    # Training configuration.
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the training.')
    
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number for Training.')
    
    parser.add_argument('--g_lr', type=float, default=0.00001, help='learning rate for G')
    
    parser.add_argument('--d_lr', type=float, default=0.00001, help='learning rate for D')
    
    parser.add_argument('--g2_lr', type=float, default=0.00001, help='learning rate for G2')
    
    parser.add_argument('--d2_lr', type=float, default=0.00001, help='learning rate for D2')
    
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    
    parser.add_argument('--dec_dropout', type=float, default=0., help='dropout rate for decoder')
    
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    
    parser.add_argument('--clipping_value', type=int, default=2, help='1,2, or 5 suggested but not strictly')
    
    parser.add_argument('--features', type=str2bool, default=False, help='features dimension for nodes')

    
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=10000, help='test model from this step')
    
    parser.add_argument('--num_test_epoch', type=int, default=30000, help='inference epoch')
    
    parser.add_argument('--inference_sample_num', type=int, default=10000, help='inference samples')
    
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    
    parser.add_argument('--inference_iterations', type=int, default=100, help='Number of iterations for inference')
    
    parser.add_argument('--inf_batch_size', type=int, default=1, help='Batch size for inference')
    
    # Directories.
    parser.add_argument('--protein_data_dir', type=str, default='DrugGEN/data/akt')      
    
    parser.add_argument('--drug_index', type=str, default='DrugGEN/data/drug_smiles.index')  
    
    parser.add_argument('--drug_data_dir', type=str, default='DrugGEN/data')    
    
    parser.add_argument('--mol_data_dir', type=str, default='DrugGEN/data')
    
    parser.add_argument('--log_dir', type=str, default='DrugGEN/experiments/logs')
    
    parser.add_argument('--model_save_dir', type=str, default='DrugGEN/experiments/models')
    
    parser.add_argument('--inference_model', type=str, default='')
    
    parser.add_argument('--sample_dir', type=str, default='DrugGEN/experiments/samples')
    
    parser.add_argument('--result_dir', type=str, default='DrugGEN/experiments/tboard_output')
    
    parser.add_argument('--dataset_file', type=str, default='chembl45_train.pt')    
    
    parser.add_argument('--drug_dataset_file', type=str, default='akt_train.pt')        
    
    parser.add_argument('--raw_file', type=str, default='DrugGEN/data/chembl_train.smi')     
      
    parser.add_argument('--drug_raw_file', type=str, default='DrugGEN/data/akt_train.smi')   
    
    parser.add_argument('--inf_dataset_file', type=str, default='chembl45_test.pt')    
    
    parser.add_argument('--inf_drug_dataset_file', type=str, default='akt_test.pt')        
    
    parser.add_argument('--inf_raw_file', type=str, default='DrugGEN/data/chembl_test.smi')     
      
    parser.add_argument('--inf_drug_raw_file', type=str, default='DrugGEN/data/akt_test.smi')
           
    # Step size.
    parser.add_argument('--log_sample_step', type=int, default=1000, help='step size for sampling during training')

    # Define the seed.
    parser.add_argument('--set_seed', type=bool, default=False, help='set seed for reproducibility')

    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility')

    # Resume training.
    parser.add_argument('--resume', type=bool, default=False, help='resume training')

    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
    
    parser.add_argument('--resume_iter', type=int, default=None, help='resume training from this step')
    
    parser.add_argument('--resume_directory', type=str, default=None, help='load pretrained weights from this directory')

    config = parser.parse_args()
    print(config)
    main(config)
