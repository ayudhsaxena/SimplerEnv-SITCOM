import os
import json
import pickle
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from google import genai
from google.genai import types
from google.genai import errors
from typing import List, Dict, Any, Tuple
import time

MAX_TRIES = 3
# Number of times to retry the API call in case of failure
class RewardMemory:
    def __init__(self, json_path: str = None, api_key: str = None, lambda_weight: float = 0.5):
        """
        Initialize the RewardMemory with a JSON file containing subtasks and image pairs.
        
        Args:
            json_path (str, optional): Path to the JSON file. If None, no data will be loaded
            api_key (str, optional): Google AI API key. If None, looks for environment variable
            lambda_weight (float): Weight for action similarity vs object similarity
        """
        # Configure the API key
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("API key must be provided either as an argument or as GOOGLE_API_KEY environment variable")
        
        self.client = genai.Client(api_key=api_key)
        self.lambda_weight = lambda_weight
        self.data = {}
        self.processed_data = {}
        
        # Set up generation model configuration for structured output
        self.generation_config = types.GenerateContentConfig(
            temperature=0.2,  # Deterministic output
            response_mime_type="application/json"  # For structured output
        )
        
        # Load the JSON data if provided
        if json_path:
            self.load_json(json_path)
    
    def load_json(self, json_path: str):
        """
        Load data from a JSON file and process it.
        
        Args:
            json_path (str): Path to the JSON file
        """
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Process the data
        self.processed_data = self._process_data()
    
    def _extract_object_action(self, subtask: str) -> Dict[str, str]:
        """
        Extract object and action from a subtask using Google's Generative AI.
        
        Args:
            subtask (str): The subtask description
            
        Returns:
            Dict with 'object' and 'action' keys
        """
        
        prompt = f"""
        Extract the action and object from the following task description.
        Return a JSON with keys 'action' and 'object'.
        
        Task: "{subtask}"
        """
        curr_try = 0
        while curr_try < MAX_TRIES:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=self.generation_config,
                )
                break
            except errors.APIError as e:
                print(f"Error: {e}. Retrying...")
                time.sleep((2 ** curr_try)*5)  # Exponential backoff
                
                curr_try += 1
                if curr_try == MAX_TRIES:
                    return {
                        'action': "",
                        'object': ""
                    }
        parsed_response = json.loads(response.text)
        
        return {
            'action': parsed_response.get('action', '') if parsed_response.get('action', '') is not None else "",
            'object': parsed_response.get('object', '') if parsed_response.get('object', '') is not None else ""
        }
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        print(f"Generating embedding for: {text}")
        curr_try = 0
        while curr_try < MAX_TRIES:
            try:
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=text
                )
                break
            except errors.APIError as e:
                print(f"Error: {e}. Retrying...")
                time.sleep((2 ** curr_try)*5)  # Exponential backoff

                curr_try += 1
                if curr_try == MAX_TRIES:
                    raise Exception(f"Failed to get embedding after {MAX_TRIES} attempts")
        
        return np.array(response.embeddings[0].values)
    
    def _process_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Process the JSON data to extract objects and actions and create embeddings.
        
        Returns:
            Dict with processed data
        """
        processed_data = {}
        
        for subtask, examples in self.data.items():
            try:
                # Extract object and action
                extracted = self._extract_object_action(subtask)
                action = extracted['action']
                obj = extracted['object']
                
                # Get embeddings
                action_embedding = self._get_embedding(action)
                object_embedding = self._get_embedding(obj)
                
                # Group examples by reward class
                reward_classes = defaultdict(list)
                for example in examples:
                    reward_class = example.get('reward_class', 0)
                    reward_classes[reward_class].append({
                        'image1_path': example.get('image1_path', ''),
                        'image2_path': example.get('image2_path', ''),
                        'raw_reward': example.get('raw_reward', 0.0),
                        'reward_class': reward_class,
                        'pair_id': example.get('pair_id', 0)
                    })
            except Exception as e:
                print(f"Error processing subtask '{subtask}': {e}")
                continue
            
            processed_data[subtask] = {
                'action': action,
                'object': obj,
                'action_embedding': action_embedding,
                'object_embedding': object_embedding,
                'reward_classes': dict(reward_classes)
            }
        
        return processed_data
    
    def _calculate_similarity(self, query_action_emb: np.ndarray, query_object_emb: np.ndarray, 
                             target_action_emb: np.ndarray, target_object_emb: np.ndarray) -> float:
        """
        Calculate similarity between query and target using object and action embeddings.
        
        Args:
            query_action_emb: Action embedding for query
            query_object_emb: Object embedding for query
            target_action_emb: Action embedding for target
            target_object_emb: Object embedding for target
            
        Returns:
            float: Combined similarity score
        """
        # Calculate cosine similarity (1 - cosine distance)
        action_sim = 1 - cosine(query_action_emb, target_action_emb)
        object_sim = 1 - cosine(query_object_emb, target_object_emb)
        
        # Combined similarity with lambda weighting
        return object_sim + self.lambda_weight * action_sim
    
    def retrieve(self, query_subtask: str, num_examples: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve similar examples for a given subtask in a stratified manner.
        
        Args:
            query_subtask (str): The subtask to find similar examples for
            num_examples (int): Number of examples to retrieve
            
        Returns:
            List of dicts containing image pairs and similarity information
        """
        # Extract object and action from query
        extracted = self._extract_object_action(query_subtask)
        query_action = extracted['action']
        query_object = extracted['object']
        
        # Get embeddings for query
        query_action_emb = self._get_embedding(query_action)
        query_object_emb = self._get_embedding(query_object)
        
        # Calculate similarity with all subtasks
        similarities = []
        for subtask, data in self.processed_data.items():
            similarity = self._calculate_similarity(
                query_action_emb, query_object_emb,
                data['action_embedding'], data['object_embedding']
            )
            similarities.append((subtask, similarity))
        
        # Sort subtasks by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create a deep copy of the reward classes to avoid modifying the original data
        subtask_data_copy = {}
        for subtask, similarity in similarities:
            subtask_data = self.processed_data[subtask]
            reward_classes_copy = {}
            for class_id, examples in subtask_data['reward_classes'].items():
                reward_classes_copy[class_id] = examples.copy()
            
            subtask_data_copy[subtask] = {
                'reward_classes': reward_classes_copy
            }
        
        # Retrieve examples in a stratified manner
        results = []
        
        # Keep track of how many examples we've collected
        collected = 0
        
        # Continue until we have enough examples or run out of data
        while collected < num_examples:
            added_in_round = 0
            
            # Go through subtasks in order of similarity
            for subtask, similarity in similarities:
                reward_classes = subtask_data_copy[subtask]['reward_classes']
                
                # Skip if no examples in this subtask
                if not reward_classes:
                    continue
                
                # Get all reward classes
                all_classes = list(reward_classes.keys())
                
                # If there are examples in this class
                if all_classes:
                    # Round-robin over reward classes
                    class_idx = collected % len(all_classes)
                    reward_class = all_classes[class_idx]
                    
                    # If there are examples in this reward class
                    if reward_classes[reward_class]:
                        # Get the first example and remove it
                        example = reward_classes[reward_class].pop(0)
                        
                        # If reward class is now empty, remove it
                        if not reward_classes[reward_class]:
                            del reward_classes[reward_class]
                        
                        # Add the example to results with similarity
                        example['subtask_similarity'] = similarity
                        example['original_subtask'] = subtask
                        results.append(example)
                        
                        collected += 1
                        added_in_round += 1
                        
                        # Break if we have enough examples
                        if collected >= num_examples:
                            break
            
            # If we didn't add any examples in this round, we've run out of data
            if added_in_round == 0:
                break
        
        return results
    
    def save(self, file_path: str):
        """
        Save the processed data and state to a file.
        
        Args:
            file_path (str): Path to save the data
        """
        # Create a dictionary with the state to save
        state = {
            'lambda_weight': self.lambda_weight,
            'processed_data': self.processed_data,
            'data': self.data
        }
        
        # Save to file using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"RewardMemory state saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str, api_key: str = None):
        """
        Load a saved RewardMemory state from a file.
        
        Args:
            file_path (str): Path to the saved state file
            api_key (str, optional): Google AI API key. If None, looks for environment variable
            
        Returns:
            RewardMemory: Initialized RewardMemory object with loaded state
        """
        # Create a new instance without loading JSON
        memory = cls(json_path=None, api_key=api_key)
        
        # Load the state from file
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        # Restore the state
        memory.lambda_weight = state.get('lambda_weight', 0.5)
        memory.processed_data = state.get('processed_data', {})
        memory.data = state.get('data', {})
        
        print(f"RewardMemory state loaded from {file_path}")
        return memory

# Example usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Example 1: Initialize, process, and save
    # memory = RewardMemory("../reward_memory_data.json", api_key=API_KEY, lambda_weight=0.7)
    
    # # Save the processed state
    # memory.save("reward_memory_state.pkl")
    
    # Example 2: Load from saved state
    loaded_memory = RewardMemory.load("reward_memory_state.pkl", api_key=API_KEY)
    
    # Use the loaded memory
    query = "Move to the block."
    results = loaded_memory.retrieve(query, num_examples=5)
    
    # Print the results
    for i, result in enumerate(results):
        print(f"Example {i+1}:")
        print(f"  Subtask similarity: {result['subtask_similarity']:.4f}")
        print(f"  Original subtask: {result['original_subtask']}")
        print(f"  Image pair: {result['pair_id']}")
        print(f"  Images: {result['image1_path']} and {result['image2_path']}")
        print(f"  Raw reward: {result['raw_reward']:.4f}")
        print(f"  Reward class: {result['reward_class']}")
        print()