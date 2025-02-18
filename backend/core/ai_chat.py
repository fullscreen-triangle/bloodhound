from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from .federated.base import FederatedModel

class AIChatModel(FederatedModel):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    async def generate_response(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate response to user query"""
        try:
            # Prepare input with context
            input_text = self._prepare_input(query, context)
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=2048,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._format_response(response)
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            raise
            
    def train_local(self, data: Dict[str, Any], epochs: int) -> None:
        """Train model on local data"""
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters())
            
            for epoch in range(epochs):
                for batch in data['train_loader']:
                    input_ids, labels = batch
                    outputs = self.model(
                        input_ids=input_ids.to(self.device),
                        labels=labels.to(self.device)
                    )
                    
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
        except Exception as e:
            logging.error(f"Error in local training: {str(e)}")
            raise
            
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for aggregation"""
        return {name: param.data for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters after aggregation"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
                    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data['val_loader']:
                input_ids, labels = batch
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    labels=labels.to(self.device)
                )
                total_loss += outputs.loss.item()
                num_batches += 1
                
        return {'avg_loss': total_loss / num_batches}
    
    def _prepare_input(self, query: str, context: Dict[str, Any] = None) -> str:
        """Prepare input text with context"""
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            return f"Context:\n{context_str}\n\nQuery: {query}"
        return query
    
    def _format_response(self, response: str) -> str:
        """Format model response"""
        # Remove any system prompts or context from response
        if "Query:" in response:
            response = response.split("Query:")[-1]
        return response.strip()

class AIChat:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.config = self._initialize_models()
        self.anthropic_client = anthropic.Client(os.getenv('ANTHROPIC_API_KEY'))
        
    def _initialize_models(self):
        """Initialize AI models based on available resources"""
        specs = self.resource_manager.get_optimal_config()
        
        if specs['model_size'] == 'tiny':
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        elif specs['model_size'] == 'small':
            model_name = "microsoft/phi-2"
        else:
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            
        # Load model based on available resources
        if specs['use_gpu']:
            device = 'cuda'
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            device = 'cpu'
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": device},
                low_cpu_mem_usage=True
            )
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device,
            'model_name': model_name
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate AI response based on query and experimental context"""
        # For complex queries, use Claude API
        if self._is_complex_query(query):
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": self._format_prompt(query, context)
                }]
            )
            return response.content
            
        # For simple queries, use local model
        else:
            inputs = self.config['tokenizer'](
                self._format_prompt(query, context),
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.config['device'])
            
            outputs = self.config['model'].generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
            
            return self.config['tokenizer'].decode(outputs[0], skip_special_tokens=True)
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if query requires more sophisticated model"""
        # Add logic to determine query complexity
        complex_indicators = [
            'compare', 'analyze', 'explain',
            'relationship between', 'why does',
            'how does', 'what is the mechanism'
        ]
        return any(indicator in query.lower() for indicator in complex_indicators)
    
    def _format_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Format prompt with experimental context"""
        return f"""Context:
Experiment ID: {context.get('id')}
Type: {context.get('type')}
Methods: {context.get('methods')}
Results: {context.get('results')}

Question: {query}

Please provide a scientific analysis based on the above context.""" 