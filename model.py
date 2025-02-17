import torch
import gc
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BertTokenizer, BertModel,
    PegasusTokenizer, PegasusForConditionalGeneration
)
import numpy as np
import time
import os
import json
from pathlib import Path
from rouge_score import rouge_scorer
from datetime import datetime

CURRENT_TIME = "2025-02-15 11:39:55"
CURRENT_USER = "Satwik-Uppada298"


class AdvancedSummarizer:
    def __init__(self, use_gpu: bool = True):
        # Memory efficient device settings
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"Using device: {self.device}")
        self.init_time = CURRENT_TIME

        # Memory-optimized configurations
        self.model_configs = {
            't5': {
                'name': 't5-base',
                'params': {
                    'max_length': 150,
                    'min_length': 60,
                    'num_beams': 4,
                    'length_penalty': 2.0,
                    'early_stopping': True,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 2.5,
                    'do_sample': True,
                    'top_k': 50,
                    'top_p': 0.92,
                    'temperature': 0.7
                }
            },
            'pegasus': {
                'name': 'google/pegasus-large',
                'params': {
                    'max_length': 150,
                    'min_length': 60,
                    'num_beams': 4,
                    'length_penalty': 2.0,
                    'early_stopping': True,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 2.5,
                    'do_sample': True,
                    'top_k': 50,
                    'top_p': 0.92,
                    'temperature': 0.7
                }
            },
            'bert': {
                'name': 'bert-base-uncased',
                'max_length': 512
            }
        }

        self.current_model = None
        self.initialize_models()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def _clear_memory(self):
        """Clear CUDA memory and garbage collect."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _load_model(self, model_type: str):
        """Load a specific model while managing memory."""
        if self.current_model == model_type:
            return

        self._clear_memory()

        if model_type == 't5':
            if hasattr(self, 't5_model'):
                self.t5_model.cpu()
                del self.t5_model
                self._clear_memory()
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                self.model_configs['t5']['name']
            ).to(self.device)
            self.t5_model.eval()

        elif model_type == 'pegasus':
            if hasattr(self, 'pegasus_model'):
                self.pegasus_model.cpu()
                del self.pegasus_model
                self._clear_memory()
            self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(
                self.model_configs['pegasus']['name']
            ).to(self.device)
            self.pegasus_model.eval()

        elif model_type == 'bert':
            if hasattr(self, 'bert_model'):
                self.bert_model.cpu()
                del self.bert_model
                self._clear_memory()
            self.bert_model = BertModel.from_pretrained(
                self.model_configs['bert']['name']
            ).to(self.device)
            self.bert_model.eval()

        self.current_model = model_type

    def initialize_models(self):
        """Initialize tokenizers only."""
        try:
            print("\nInitializing tokenizers...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                self.model_configs['t5']['name'],
                model_max_length=1024
            )
            self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(
                self.model_configs['pegasus']['name']
            )
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.model_configs['bert']['name']
            )
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text."""
        text = ' '.join(text.split())
        sentences = []
        for sent in text.split('.'):
            sent = sent.strip()
            if sent:
                if not sent[0].isupper():
                    sent = sent.capitalize()
                sentences.append(sent)
        text = '. '.join(sentences)
        if not text.endswith('.'):
            text += '.'
        return text

    def get_bert_embeddings(self, sentences: list) -> torch.Tensor:
        """Generate BERT embeddings with memory optimization."""
        embeddings = []
        batch_size = 4  # Process in smaller batches

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.bert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.model_configs['bert']['max_length'],
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(embedding.cpu())  # Move to CPU to save GPU memory

        return torch.cat(embeddings, dim=0).to(self.device)

    def select_best_sentences(self, sentences: list, embeddings: torch.Tensor) -> list:
        """Select top sentences based on BERT embeddings."""
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        importance_scores = similarity_matrix.sum(dim=1)
        _, indices = torch.topk(importance_scores, min(4, len(sentences)))
        return [sentences[idx] for idx in sorted(indices.tolist())]

    def generate_model_summary(self, text: str, model_type: str) -> tuple:
        """Generate summary using specified model with memory management."""
        start_time = time.time()
        try:
            self._load_model(model_type)

            if model_type == 't5':
                inputs = self.t5_tokenizer(
                    "summarize: " + text,
                    max_length=1024,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **self.model_configs['t5']['params']
                    )
                summary = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif model_type == 'pegasus':
                inputs = self.pegasus_tokenizer(
                    text,
                    max_length=1024,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.pegasus_model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        **self.model_configs['pegasus']['params']
                    )
                summary = self.pegasus_tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif model_type == 'bert':
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                embeddings = self.get_bert_embeddings(sentences)
                selected_sentences = self.select_best_sentences(sentences, embeddings)
                summary = '. '.join(selected_sentences) + '.'

            summary = self.preprocess_text(summary)
            return summary, time.time() - start_time

        except Exception as e:
            print(f"Error generating summary with {model_type}: {str(e)}")
            return f"Error with {model_type} summary generation.", time.time() - start_time

    def create_ensemble_summary(self, summaries: dict) -> str:
        """Create enhanced ensemble summary."""
        all_sentences = {}
        model_weights = {
            'pegasus': 1.4,
            't5': 1.2,
            'bert': 1.0
        }

        for model, summary in summaries.items():
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            for sent in sentences:
                sent_lower = sent.lower()
                if sent_lower not in all_sentences:
                    all_sentences[sent_lower] = {
                        'original': sent,
                        'score': 0,
                        'models': []
                    }
                all_sentences[sent_lower]['score'] += model_weights[model]
                all_sentences[sent_lower]['models'].append(model)

        ranked_sentences = sorted(
            all_sentences.values(),
            key=lambda x: (x['score'], len(x['models']), len(x['original'].split())),
            reverse=True
        )

        final_sentences = []
        total_words = 0
        max_words = 100

        for item in ranked_sentences:
            sentence_words = len(item['original'].split())
            if total_words + sentence_words <= max_words:
                final_sentences.append(item['original'])
                total_words += sentence_words
            if total_words >= max_words:
                break

        summary = '. '.join(final_sentences)
        if not summary.endswith('.'):
            summary += '.'
        return summary

    def calculate_rouge_scores(self, reference: str, summary: str) -> dict:
        """Calculate ROUGE scores with optimized weights."""
        scores = self.rouge_scorer.score(reference, summary)
        weighted_score = (
                scores['rouge1'].fmeasure * 0.35 +
                scores['rouge2'].fmeasure * 0.45 +
                scores['rougeL'].fmeasure * 0.20
        )
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'weighted': weighted_score
        }

    def summarize(self, text: str) -> dict:
        """Generate summaries using all models and create ensemble."""
        try:
            text = self.preprocess_text(text)
            start_time = time.time()

            summaries = {}
            times = {}
            scores = {}

            for model_type in ['t5', 'pegasus', 'bert']:
                summary, gen_time = self.generate_model_summary(text, model_type)
                summaries[model_type] = summary
                times[model_type] = gen_time
                scores[model_type] = self.calculate_rouge_scores(text, summary)
                self._clear_memory()  # Clear memory after each model

            ensemble_summary = self.create_ensemble_summary(summaries)
            ensemble_scores = self.calculate_rouge_scores(text, ensemble_summary)

            return {
                't5': {
                    'summary': summaries['t5'],
                    'scores': scores['t5'],
                    'time': times['t5']
                },
                'pegasus': {
                    'summary': summaries['pegasus'],
                    'scores': scores['pegasus'],
                    'time': times['pegasus']
                },
                'bert': {
                    'summary': summaries['bert'],
                    'scores': scores['bert'],
                    'time': times['bert']
                },
                'ensemble': {
                    'summary': ensemble_summary,
                    'scores': ensemble_scores,
                    'time': time.time() - start_time
                }
            }

        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return None

    def save_models(self, save_dir="models"):
        """Save all models and configurations."""
        try:
            timestamp = CURRENT_TIME.replace(" ", "_").replace(":", "-")
            save_path = Path(save_dir) / f"version_{timestamp}"
            save_path.mkdir(parents=True, exist_ok=True)

            # Save metadata
            metadata = {
                "timestamp": CURRENT_TIME,
                "user": CURRENT_USER,
                "model_configs": self.model_configs
            }

            with open(save_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)

            # Save models
            print("\nSaving models...")

            # Save T5
            print("Saving T5...")
            t5_path = save_path / "t5-base"
            t5_path.mkdir(exist_ok=True)
            self._load_model('t5')
            self.t5_model.save_pretrained(str(t5_path))
            self.t5_tokenizer.save_pretrained(str(t5_path))
            self._clear_memory()

            # Save PEGASUS
            print("Saving PEGASUS...")
            pegasus_path = save_path / "pegasus-large"
            pegasus_path.mkdir(exist_ok=True)
            self._load_model('pegasus')
            self.pegasus_model.save_pretrained(str(pegasus_path))
            self.pegasus_tokenizer.save_pretrained(str(pegasus_path))
            self._clear_memory()

            # Save BERT
            print("Saving BERT...")
            bert_path = save_path / "bert-base-uncased"
            bert_path.mkdir(exist_ok=True)
            self._load_model('bert')
            self.bert_model.save_pretrained(str(bert_path))
            self.bert_tokenizer.save_pretrained(str(bert_path))
            self._clear_memory()

            print(f"\nAll models saved successfully in: {save_path}")
            return str(save_path)

        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise


def main():
    print(f"Current time (UTC): {CURRENT_TIME}")
    print(f"Current user: {CURRENT_USER}")

    summarizer = AdvancedSummarizer(use_gpu=True)

    test_text = """
    Artificial intelligence (AI) has transformed various sectors, from healthcare to transportation.
    Machine learning algorithms now power everything from medical diagnosis to self-driving cars.
    Deep learning networks have achieved unprecedented accuracy in image and speech recognition.
    However, concerns about AI ethics and bias remain significant challenges.
    Researchers worldwide are working to develop more transparent and accountable AI systems.
    The future of AI depends on balancing innovation with responsible development practices.
    """

    results = summarizer.summarize(test_text)

    if results:
        print("\nSummary Results:")
        print("-" * 60)

        for model_name in ['t5', 'pegasus', 'bert', 'ensemble']:
            print(f"\n{model_name.upper()} Summary:")
            print(results[model_name]['summary'])
            print("\nPerformance Metrics:")
            print(f"ROUGE-1: {results[model_name]['scores']['rouge1']:.4f}")
            print(f"ROUGE-2: {results[model_name]['scores']['rouge2']:.4f}")
            print(f"ROUGE-L: {results[model_name]['scores']['rougeL']:.4f}")
            print(f"Weighted Score: {results[model_name]['scores']['weighted']:.4f}")
            print(f"Generation Time: {results[model_name]['time']:.2f}s")
            print("-" * 60)

    # Save the models
    print("\nSaving models...")
    save_path = summarizer.save_models()
    print(f"Models saved successfully in: {save_path}")


if __name__ == "__main__":
    main()