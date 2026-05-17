"""
Mutation Service - Adversarial mutation integration.

Interfaces with Red-Agent v1 mutation framework.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class MutationService:
    """Service for generating and managing adversarial mutations."""
    
    def __init__(self):
        """Initialize mutation service."""
        self.red_agent_path = Path(settings.CYBERSHIELD_SRC) / "red_agent"
        self._mutation_registry = None
        self._is_initialized = False
    
    def initialize(self):
        """Initialize mutation framework."""
        if self._is_initialized:
            return
            
        try:
            from src.red_agent.mutation_registry import get_mutation_registry
            self._mutation_registry = get_mutation_registry()
            
            logger.info(f"Mutation registry loaded with {len(self._mutation_registry)} operators")
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize mutation service: {e}")
            raise
    
    def list_mutations(self) -> Dict[str, Dict[str, Any]]:
        """List available mutations."""
        if not self._is_initialized:
            self.initialize()
            
        try:
            mutations_info = {}
            for name, operator_class in self._mutation_registry.items():
                mutations_info[name] = {
                    "name": name,
                    "description": operator_class.__doc__ or "No description",
                    "severity_range": [0.0, 1.0],
                    "parameters": {}  # Can be extended per operator
                }
            return mutations_info
            
        except Exception as e:
            logger.error(f"Failed to list mutations: {e}")
            raise
    
    def apply_mutation(
        self,
        session_data: np.ndarray,
        mutation_type: str,
        severity: float = 0.5,
        seed: int = 42,
        **params
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply a mutation to session data.
        
        Args:
            session_data: Session feature array
            mutation_type: Name of mutation operator
            severity: Mutation severity [0, 1]
            seed: Random seed for reproducibility
            **params: Mutation-specific parameters
            
        Returns:
            Tuple of (mutated_data, mutation_metadata)
        """
        if not self._is_initialized:
            self.initialize()
            
        try:
            # Get mutation operator
            if mutation_type not in self._mutation_registry:
                raise ValueError(f"Unknown mutation type: {mutation_type}")
                
            operator_class = self._mutation_registry[mutation_type]
            
            # Create operator instance
            operator = operator_class()
            
            # Create mutation request
            from src.red_agent.mutation_engine.contracts import MutationRequest
            mutation_req = MutationRequest(
                mutation_type=mutation_type,
                severity=severity,
                parameters=params
            )
            
            # Apply mutation
            mutation_result = operator.mutate(session_data, mutation_req, seed=seed)
            
            # Extract metadata
            metadata = {
                "mutation_type": mutation_type,
                "severity": severity,
                "seed": seed,
                "parameters": params,
                "applied_at_step": getattr(mutation_result, "applied_at_step", 0),
            }
            
            return mutation_result.mutated_data, metadata
            
        except Exception as e:
            logger.error(f"Mutation application failed: {e}")
            raise
    
    def batch_mutate(
        self,
        dataset_path: Path,
        mutations: List[Dict[str, Any]],
        base_seed: int = 42,
        sample_limit: int = None,
        batch_size: int = 256
    ) -> Tuple[Path, Path]:
        """
        Apply mutations to entire dataset.
        
        Returns:
            Tuple of (mutated_npz_path, metadata_jsonl_path)
        """
        try:
            # Load dataset
            data = np.load(dataset_path)
            sessions = data['sessions']
            labels = data['labels'] if 'labels' in data else np.ones(len(sessions))
            
            if sample_limit:
                sessions = sessions[:sample_limit]
                labels = labels[:sample_limit]
            
            logger.info(f"Loaded {len(sessions)} sessions from {dataset_path}")
            
            # Apply mutations
            mutated_sessions = []
            metadata_list = []
            
            for idx, session in enumerate(sessions):
                mutated_session = session.copy()
                
                for mutation_spec in mutations:
                    try:
                        mutated_session, meta = self.apply_mutation(
                            mutated_session,
                            mutation_type=mutation_spec['mutation_type'],
                            severity=mutation_spec.get('severity', 0.5),
                            seed=base_seed + idx,
                            **mutation_spec.get('params', {})
                        )
                        meta['original_label'] = int(labels[idx])
                        metadata_list.append(meta)
                        
                    except Exception as e:
                        logger.warning(f"Mutation {mutation_spec['mutation_type']} failed for sample {idx}: {e}")
                        continue
                
                mutated_sessions.append(mutated_session)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} samples")
            
            # Save mutated dataset
            mutated_sessions = np.array(mutated_sessions, dtype=np.float32)
            output_dir = Path(settings.ARTIFACTS_DIR)
            output_dir.mkdir(exist_ok=True)
            
            mutated_path = output_dir / "mutated_sessions.npz"
            np.savez_compressed(
                mutated_path,
                sessions=mutated_sessions,
                labels=labels
            )
            logger.info(f"Saved mutated dataset to {mutated_path}")
            
            # Save metadata
            metadata_path = output_dir / "mutation_metadata.jsonl"
            with open(metadata_path, 'w') as f:
                for meta in metadata_list:
                    f.write(json.dumps(meta) + '\n')
            logger.info(f"Saved metadata to {metadata_path}")
            
            return mutated_path, metadata_path
            
        except Exception as e:
            logger.error(f"Batch mutation failed: {e}")
            raise
