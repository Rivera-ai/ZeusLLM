import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import math
from tqdm import tqdm
from llm import Transformer
import torch.nn.functional as F

class LLMInference:
    def __init__(self, checkpoint_path="outputs/best_model.pt", device=None):
        try:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
            
            print(f"Dispositivo: {self.device}")
            print(f"Cargando modelo desde {checkpoint_path}")
            
            # Cargar checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                print("Checkpoint cargado exitosamente")
            except Exception as e:
                print(f"Error al cargar checkpoint: {e}")
                raise
            
            # Obtener y mostrar argumentos del modelo
            self.model_args = checkpoint['model_args']
            print("\nArgumentos del modelo:")
            for key, value in vars(self.model_args).items():
                print(f"{key}: {value}")
            
            # Inicializar modelo con más información
            print("\nInicializando modelo...")
            self.model = Transformer(self.model_args)
            print("Modelo creado")
            
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Estado del modelo cargado")
            except Exception as e:
                print(f"Error al cargar estado del modelo: {e}")
                raise
                
            self.model.to(self.device)
            self.model.eval()
            
            print("\nInicializando tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3.1-8B",
                    model_max_length=self.model_args.max_seq_len,
                    padding_side='left',
                    truncation_side='left',
                    bos_token='<s>',
                    eos_token='</s>',
                    unk_token='<unk>',
                    use_fast=True
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Tokenizer inicializado")
            except Exception as e:
                print(f"Error al inicializar tokenizer: {e}")
                raise
            
            print("\nInformación del modelo:")
            print(f"Tamaño de vocabulario del tokenizer: {self.tokenizer.vocab_size}")
            print(f"Tamaño de vocabulario del modelo: {self.model_args.vocab_size}")
            print(f"Tokens especiales: {self.tokenizer.special_tokens_map}")
            print(f"Longitud máxima de secuencia: {self.model_args.max_seq_len}")
            
        except Exception as e:
            print(f"Error en inicialización: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1
    ):
        try:
            print("\nInformación de generación:")
            print(f"Prompt: {prompt}")
        
        # Tokenizar correctamente
            encoding = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.model_args.max_seq_len - max_new_tokens,
                return_tensors='pt'
            )
        
            input_ids = encoding['input_ids'].to(self.device)
            print(f"Tokens codificados: {input_ids.tolist()}")
            print(f"Shape de input_ids: {input_ids.shape}")
        
        # Verificar que no hay tokens fuera de rango
            if torch.any(input_ids >= self.model_args.vocab_size):
                print("Advertencia: Tokens fuera de rango detectados")
                input_ids = torch.clamp(input_ids, 0, self.model_args.vocab_size - 1)
        
            generated = input_ids[0].tolist()
        
            print("\nIniciando generación token por token...")
            for i in range(max_new_tokens):
                try:
                # Preparar input actual
                    curr_input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
                    #print(f"\nPaso {i+1}:")
                    #print(f"Shape actual: {curr_input_ids.shape}")
                
                    if curr_input_ids.shape[1] >= self.model_args.max_seq_len:
                        print("Longitud máxima alcanzada")
                        break
                
                # Obtener predicciones
                    outputs = self.model(curr_input_ids)
                
                # Obtener logits del último token
                    logits = outputs[:, -1, :].float()
                
                # Aplicar temperatura
                    logits = logits / temperature
                
                # Top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                        logits[logits < v[:, [-1]]] = float('-inf')
                
                # Aplicar softmax
                    probs = F.softmax(logits, dim=-1)
                
                # Muestrear siguiente token
                    next_token = torch.multinomial(probs, num_samples=1)[0]
                
                # Verificar que el token está en rango
                    if next_token.item() >= self.model_args.vocab_size:
                        print(f"Token fuera de rango generado: {next_token.item()}")
                        next_token = torch.tensor(self.tokenizer.eos_token_id, device=self.device)
                
                    generated.append(next_token.item())
                    #print(f"Token generado: {next_token.item()}")
                
                # Si encontramos EOS, terminar
                    if next_token.item() == self.tokenizer.eos_token_id:
                        print("Token EOS encontrado")
                        break
                
                except Exception as e:
                    print(f"Error en paso de generación {i+1}: {e}")
                    raise
        
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            print(f"\nTexto generado: {result}")
            return result
        
        except Exception as e:
            print(f"Error en generate(): {e}")
            import traceback
            print(f"Traceback completo:\n{traceback.format_exc()}")
            return None

def main():
    # Inicializar generador
    generator = LLMInference("outputs/best_model.pt")
    
    prompts = [
        "En un futuro lejano,",
        "El sentido de la vida es",
        "La inteligencia artificial",
        "Las mejores prácticas para programar incluyen"
    ]
    
    print("=== Probando el modelo ===\n")
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generación:")
        try:
            response = generator.generate(
                prompt,
                max_new_tokens=20,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1
            )
            if response:
                print(f"\nRespuesta completa: {response}\n")
        except Exception as e:
            print(f"Error al generar respuesta: {str(e)}")
        print("-" * 50)

if __name__ == "__main__":
    main()