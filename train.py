import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import math
from tqdm import tqdm
from llm import Transformer, ModelArgs
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard.writer import SummaryWriter

class TokenizedDataset(Dataset):
    def __init__(self, data_path, meta_path, max_length=2048):
        self.data_path = Path(data_path)
        self.max_length = max_length
        
        # Cargar metadatos
        self.meta = np.load(meta_path, allow_pickle=True).item()
        
        # Cargar datos tokenizados
        self.data = np.memmap(
            str(data_path),
            dtype=np.uint16,
            mode='r',
            shape=(self.meta['total_tokens'],)
        )
        
        # Calcular número de secuencias completas
        self.num_sequences = math.floor(len(self.data) / self.max_length)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        
        sequence = self.data[start_idx:end_idx].astype(np.int64)
        return {
            'input_ids': torch.tensor(sequence[:-1], dtype=torch.long),
            'labels': torch.tensor(sequence[1:], dtype=torch.long),
        }

def create_dataloaders(batch_size=4):
    train_dataset = TokenizedDataset(
        "data/train.bin",
        "data/train_meta.npy"
    )
    
    val_dataset = TokenizedDataset(
        "data/val.bin",
        "data/val_meta.npy"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, writer):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            _, loss = model(input_ids, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Logging cada 10 batches
        if batch_idx % 10 == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        with autocast():
            _, loss = model(input_ids, labels)
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar TensorBoard
    writer = SummaryWriter(output_dir / "logs")
    
    # Cargar metadatos para obtener vocab_size
    train_meta = np.load("data/train_meta.npy", allow_pickle=True).item()
    
    # Inicializar modelo
    model_args = ModelArgs(
        dim=768,  
        n_layers=12,  
        n_heads=12,
        n_kv_heads=12,
        vocab_size=train_meta['vocab_size'],
        max_seq_len=2048,
        dropout=0.1
    )
    
    model = Transformer(model_args)
    model = model.to(device)
    
    # Optimizador y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # Crear dataloaders
    train_loader, val_loader = create_dataloaders(batch_size=4)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("Iniciando entrenamiento...")
    print(f"Dispositivo: {device}")
    print(f"Batches por época: {len(train_loader)}")
    
    for epoch in range(3):
        print(f"\nÉpoca {epoch + 1}/3")
        
        # Entrenamiento
        train_loss = train_epoch(
            model, train_loader, optimizer, None, scaler, device, epoch, writer
        )
        
        # Evaluación
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Logging en TensorBoard
        writer.add_scalars('Loss/epoch', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Nuevo mejor modelo encontrado! (Val Loss: {val_loss:.4f})")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_args': model_args,
            }, output_dir / 'best_model.pt')
    
    writer.close()
    print("\nEntrenamiento completado!")
    print(f"Mejor pérdida de validación: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()