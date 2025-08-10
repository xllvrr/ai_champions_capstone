from typing import Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json
from pathlib import Path


class RAGEvaluator:
    """Basic evaluation framework for RAG systems."""
    
    def __init__(self, embeddings_model: OpenAIEmbeddings = None, llm_model: str = "gpt-4o-mini"):
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.llm = init_chat_model(llm_model, temperature=0)
    
    def evaluate_retrieval_relevance(self, query: str, retrieved_docs: list, top_k: int = 5) -> dict[str, Any]:
        """Evaluate relevance of retrieved documents using embedding similarity."""
        if not retrieved_docs:
            return {"relevance_scores": [], "mean_relevance": 0.0, "top_k_relevance": 0.0}
        
        # Vectorized embedding computation
        query_embedding = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
        
        # Get embeddings for all documents at once for efficiency
        doc_texts = [doc.page_content for doc in retrieved_docs]
        doc_embeddings = np.array(self.embeddings.embed_documents(doc_texts))
        
        # Vectorized cosine similarity computation
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Calculate metrics
        mean_relevance = float(np.mean(similarities))
        top_k_relevance = float(np.mean(similarities[:min(top_k, len(similarities))]))
        
        return {
            "relevance_scores": similarities.tolist(),
            "mean_relevance": mean_relevance,
            "top_k_relevance": top_k_relevance,
            "num_docs": len(retrieved_docs)
        }
    
    def evaluate_answer_quality(self, question: str, answer: str, context: str) -> dict[str, Any]:
        """Evaluate answer quality using LLM-based scoring."""
        evaluation_prompt = ChatPromptTemplate.from_template("""
        You are an expert evaluator. Rate the quality of this answer based on the given context and question.
        
        Question: {question}
        Context: {context}
        Answer: {answer}
        
        Evaluate on these criteria (1-5 scale):
        1. Relevance: How well does the answer address the question?
        2. Accuracy: How accurate is the information based on the context?
        3. Completeness: How complete is the answer?
        4. Clarity: How clear and well-structured is the answer?
        
        Respond with ONLY a JSON object:
        {{
            "relevance": <score>,
            "accuracy": <score>,
            "completeness": <score>,
            "clarity": <score>,
            "overall": <average_score>,
            "reasoning": "<brief explanation>"
        }}
        """)
        
        try:
            response = self.llm.invoke(
                evaluation_prompt.format(question=question, answer=answer, context=context)
            )
            
            # Extract content from response
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
            
            # Parse JSON response
            scores = json.loads(content.strip())
            return scores
            
        except Exception as e:
            print(f"Answer quality evaluation failed: {e}")
            return {
                "relevance": 0,
                "accuracy": 0,
                "completeness": 0,
                "clarity": 0,
                "overall": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate_source_attribution(self, answer: str, sources: list[dict]) -> dict[str, Any]:
        """Evaluate how well the answer attributes information to sources."""
        if not sources:
            return {"attribution_score": 0.0, "sources_used": 0, "total_sources": 0}
        
        # Simple heuristic: count source mentions in answer
        source_mentions = 0
        total_sources = len(sources)
        
        # Vectorized source checking
        source_identifiers = [
            f"{src.get('source', '')}_page_{src.get('page', '')}" 
            for src in sources
        ]
        
        answer_lower = answer.lower()
        mentioned_sources = [
            1 for identifier in source_identifiers
            if any(part.lower() in answer_lower for part in identifier.split('_') if part)
        ]
        
        source_mentions = sum(mentioned_sources)
        attribution_score = source_mentions / total_sources if total_sources > 0 else 0.0
        
        return {
            "attribution_score": attribution_score,
            "sources_used": source_mentions,
            "total_sources": total_sources
        }
    
    def comprehensive_evaluation(
        self, 
        question: str, 
        answer: str, 
        context: str, 
        retrieved_docs: list,
        sources: list[dict]
    ) -> dict[str, Any]:
        """Run comprehensive evaluation of the RAG system."""
        
        # Vectorized evaluation of all components
        retrieval_eval = self.evaluate_retrieval_relevance(question, retrieved_docs)
        answer_eval = self.evaluate_answer_quality(question, answer, context)
        attribution_eval = self.evaluate_source_attribution(answer, sources)
        
        # Calculate overall score
        overall_score = np.mean([
            retrieval_eval.get("top_k_relevance", 0) * 5,  # Convert to 1-5 scale
            answer_eval.get("overall", 0),
            attribution_eval.get("attribution_score", 0) * 5  # Convert to 1-5 scale
        ])
        
        return {
            "overall_score": float(overall_score),
            "retrieval": retrieval_eval,
            "answer_quality": answer_eval,
            "source_attribution": attribution_eval,
            "timestamp": str(Path.cwd()),
            "question": question[:100] + "..." if len(question) > 100 else question
        }


class EvaluationDataset:
    """Simple evaluation dataset manager."""
    
    def __init__(self, dataset_path: str = "evaluation_results.json"):
        self.dataset_path = dataset_path
        self.results = self._load_results()
    
    def _load_results(self) -> list[dict]:
        """Load existing evaluation results."""
        try:
            with open(self.dataset_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_evaluation(self, evaluation_result: dict):
        """Save evaluation result to dataset."""
        self.results.append(evaluation_result)
        with open(self.dataset_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary statistics of evaluations."""
        if not self.results:
            return {"message": "No evaluations available"}
        
        # Vectorized computation of summary statistics
        overall_scores = [r.get("overall_score", 0) for r in self.results]
        retrieval_scores = [r.get("retrieval", {}).get("top_k_relevance", 0) for r in self.results]
        answer_scores = [r.get("answer_quality", {}).get("overall", 0) for r in self.results]
        
        return {
            "total_evaluations": len(self.results),
            "average_overall_score": float(np.mean(overall_scores)),
            "average_retrieval_score": float(np.mean(retrieval_scores)),
            "average_answer_score": float(np.mean(answer_scores)),
            "score_std": float(np.std(overall_scores)),
            "latest_score": overall_scores[-1] if overall_scores else 0
        }