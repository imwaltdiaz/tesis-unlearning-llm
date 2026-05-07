# open-unlearning/src/trainer/ssu_trainer.py
from transformers import Trainer
import torch

class SSUTrainer(Trainer):
    """
    Porta SSU (Dou et al. 2024) al framework OpenUnlearning.
    Combina: task vectors + random labeling loss + weight saliency
    """
    
    def __init__(self, *args, epsilon1=1.0, epsilon2=0.5, gamma_threshold=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.gamma_threshold = gamma_threshold  # umbral saliency
        self.theta_original = None  # guarda pesos originales para task vector
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Loss 1: gradiente descendente sobre D_forget
        outputs = model(**inputs)
        L_fgt = outputs.loss
        
        # Loss 2: random labeling (mezcla inputs con outputs aleatorios)
        L_rnd = self._random_labeling_loss(model, inputs)
        
        total_loss = self.epsilon1 * L_fgt + self.epsilon2 * L_rnd
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _random_labeling_loss(self, model, inputs):
        """Mezcla aleatoriamente los labels del batch"""
        shuffled_inputs = inputs.copy()
        idx = torch.randperm(inputs['labels'].size(0))
        shuffled_inputs['labels'] = inputs['labels'][idx]
        outputs = model(**shuffled_inputs)
        return outputs.loss
    
    def _compute_weight_saliency(self, model):
        """Máscara de saliency basada en gradientes"""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.abs()
        
        # Threshold: 1 std dev sobre la media
        all_grads = torch.cat([g.flatten() for g in gradients.values()])
        threshold = all_grads.mean() + self.gamma_threshold * all_grads.std()
        
        masks = {name: (g >= threshold).float() 
                for name, g in gradients.items()}
        return masks
    
    def training_step(self, model, inputs):
        """Override para aplicar weight saliency durante el update"""
        loss = self.compute_loss(model, inputs)
        loss.backward()
        
        # Aplica saliency mask
        masks = self._compute_weight_saliency(model)
        for name, param in model.named_parameters():
            if name in masks and param.grad is not None:
                param.grad *= masks[name]
        
        return loss.detach()
    
    def apply_task_vector(self, model, theta_finetuned, theta_original):
        """
        θ_u = θ_original - (θ_finetuned - θ_original)
        Aplica al final de cada time step
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in theta_finetuned and name in theta_original:
                    task_vector = theta_finetuned[name] - theta_original[name]
                    param.data = theta_original[name] - task_vector
        return model

# Registra en OpenUnlearning
from openunlearning.trainer import _register_trainer
_register_trainer(SSUTrainer)