import requests
import json
import os
import hashlib
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class KnowledgeIntegrator:
    """Integrates external knowledge from Wikidata and ConceptNet."""
    
    def __init__(self, wikidata_endpoint: str = "https://query.wikidata.org/sparql", cache_dir: str = ".cmvkg_cache"):
        self.wikidata_endpoint = wikidata_endpoint
        self.session = requests.Session()
        self.headers = {'User-Agent': 'CMVKG-Guard-Bot/1.0', "Accept": "application/sparql-results+json"}
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, prefix: str, term: str) -> str:
        term_hash = hashlib.md5(term.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{prefix}_{term_hash}.json")

    def _load_cache(self, prefix: str, term: str) -> Optional[Any]:
        path = self._get_cache_path(prefix, term)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None
        
    def _save_cache(self, prefix: str, term: str, data: Any):
        path = self._get_cache_path(prefix, term)
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def query_wikidata(self, entity_name: str) -> Dict[str, Any]:
        """
        Queries Wikidata for properties (instance of, subclass of, part of) of a given entity.
        """
        cached = self._load_cache("wikidata", entity_name)
        if cached is not None:
            return cached
            
        # Refined SPARQL query to search for item by label and get P31, P279, P361
        sparql_query = f"""
        SELECT ?propLabel ?valLabel WHERE {{
          ?item rdfs:label "{entity_name}"@en.
          VALUES ?prop {{ wdt:P31 wdt:P279 wdt:P361 }}
          ?item ?prop ?val.
          SERVICE wikibase:label {{ 
             bd:serviceParam wikibase:language "en". 
             ?prop rdfs:label ?propLabel.
             ?val rdfs:label ?valLabel.
          }}
        }} LIMIT 15
        """
        
        results_dict = {}
        try:
            response = self.session.get(
                self.wikidata_endpoint, 
                params={'format': 'json', 'query': sparql_query},
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            
            for binding in data.get('results', {}).get('bindings', []):
                prop = binding.get('propLabel', {}).get('value', '')
                val = binding.get('valLabel', {}).get('value', '')
                if prop and val:
                    if prop not in results_dict:
                        results_dict[prop] = []
                    if val not in results_dict[prop]:
                        results_dict[prop].append(val)
                        
            self._save_cache("wikidata", entity_name, results_dict)
            return results_dict
        except Exception as e:
            logger.error(f"Wikidata query failed for {entity_name}: {e}")
            return {}

    def query_conceptnet(self, term: str) -> List[Dict[str, Any]]:
        """Queries ConceptNet for related terms."""
        cached = self._load_cache("conceptnet", term)
        if cached is not None:
            return cached
            
        url = f"http://api.conceptnet.io/c/en/{term.replace(' ', '_')}"
        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            edges = []
            for edge in data.get('edges', [])[:10]: # Limit to 10
                 rel = edge.get('rel', {}).get('label')
                 start = edge.get('start', {}).get('label')
                 end = edge.get('end', {}).get('label')
                 if rel and start and end: 
                     edges.append({
                         'rel': rel,
                         'start': start,
                         'end': end,
                         'weight': edge.get('weight', 1.0)
                     })
            self._save_cache("conceptnet", term, edges)
            return edges
        except Exception as e:
            logger.error(f"ConceptNet query failed for {term}: {e}")
            # Fallback for demo purposes if API is down
            fallback_data = {
                "cat": [{"rel": "IsA", "start": "cat", "end": "animal", "weight": 2.0},
                        {"rel": "AtLocation", "start": "cat", "end": "house", "weight": 1.0}],
                "remote": [{"rel": "UsedFor", "start": "remote", "end": "control", "weight": 2.0}],
                "couch": [{"rel": "AtLocation", "start": "couch", "end": "living room", "weight": 2.0}],
                 "flying": [{"rel": "IsA", "start": "flying", "end": "motion", "weight": 1.0}]
            }
            if term.lower() in fallback_data:
                logger.info(f"Using fallback data for '{term}'")
                return fallback_data[term.lower()]
            return []

    def enrich_entity(self, entity_label: str) -> Dict[str, Any]:
        """Enriches an entity with combined knowledge."""
        conceptnet_data = self.query_conceptnet(entity_label)
        wikidata_data = self.query_wikidata(entity_label)
        
        return {
            "entity": entity_label,
            "conceptnet_relations": conceptnet_data,
            "wikidata_properties": wikidata_data
        }
