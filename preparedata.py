import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer

class DataPreparation:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        max_seq_length: int = 2048,
        num_proc: int = 8,
        output_dir: str = "data",
        val_size: float = 0.0005,
    ):
        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        self.output_dir = Path(output_dir)
        self.val_size = val_size
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Inicializar tokenizador usando AutoTokenizer
        self.init_tokenizer(model_name)
    
    def init_tokenizer(self, model_name: str):
        """
        Inicializa el tokenizer desde el modelo pre-entrenado
        """
        print(f"Cargando tokenizer desde {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
    
    def process(self, example):
        """Procesar un ejemplo del dataset"""
        tokens = self.tokenizer.encode(
            example['text'],
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length
        )
        
        return {
            'ids': tokens,
            'len': len(tokens)
        }
    
    def prepare_dataset(self):
        """Preparar el dataset completo"""
        print("Cargando dataset...")
        dataset = load_dataset("openwebtext", num_proc=self.num_proc)
        dataset = dataset['train'].select(range(8197))
        
        # Split train/val
        split_dataset = dataset.train_test_split(
            test_size=self.val_size,
            seed=2357,
            shuffle=True
        )
        split_dataset['val'] = split_dataset.pop('test')
        
        print("Tokenizando datos...")
        tokenized = split_dataset.map(
            self.process,
            remove_columns=['text'],
            desc="Tokenizando",
            num_proc=self.num_proc
        )
        
        # Guardar en archivos binarios
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = self.output_dir / f'{split}.bin'
            
            print(f"Guardando {split}.bin...")
            dtype = np.uint16
            arr = np.memmap(str(filename), dtype=dtype, mode='w+', shape=(arr_len,))
            
            idx = 0
            
            if len(dset) < 1024:
                batch = dset.with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx:idx + len(arr_batch)] = arr_batch
            else:
                # Procesar en shards para datasets grandes
                total_batches = 1024
                for batch_idx in tqdm(range(total_batches), desc=f'Escribiendo {filename}'):
                    batch = dset.shard(
                        num_shards=total_batches,
                        index=batch_idx,
                        contiguous=True
                    ).with_format('numpy')
                
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx:idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
            
            arr.flush()
            
            # Guardar metadatos
            meta = {
                'total_tokens': arr_len,
                'vocab_size': self.vocab_size,
                'max_seq_length': self.max_seq_length,
                'tokenizer_config': {
                    'vocab_size': self.tokenizer.vocab_size,
                    'max_length': self.max_seq_length,
                    'special_tokens': self.tokenizer.special_tokens_map
                }
            }
            np.save(str(self.output_dir / f'{split}_meta.npy'), meta)
            
        print("\nEstadísticas del dataset:")
        print(f"Tamaño de vocabulario: {self.vocab_size}")
        print(f"Longitud máxima de secuencia: {self.max_seq_length}")
        for split in ['train', 'val']:
            meta = np.load(str(self.output_dir / f'{split}_meta.npy'), allow_pickle=True).item()
            total_tokens = meta['total_tokens']
            print(f"\n{split.capitalize()}:")
            print(f"Total tokens: {total_tokens:,}")
            print(f"Tamaño archivo: {os.path.getsize(self.output_dir / f'{split}.bin') / 1e9:.2f} GB")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preparar dataset para entrenamiento")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B", help="Nombre del modelo pre-entrenado")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Longitud máxima de secuencia")
    parser.add_argument("--num-proc", type=int, default=8, help="Número de procesos")
    parser.add_argument("--output-dir", type=str, default="data", help="Directorio de salida")
    parser.add_argument("--val-size", type=float, default=0.0005, help="Tamaño del conjunto de validación")
    
    args = parser.parse_args()
    
    data_prep = DataPreparation(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        num_proc=args.num_proc,
        output_dir=args.output_dir,
        val_size=args.val_size
    )
    
    data_prep.prepare_dataset()

if __name__ == '__main__':
    main()