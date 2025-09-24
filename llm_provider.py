#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Any, List, Union
import time
import re
import json
from utils.logger import logger

# Import LLM libraries - commented by default to avoid dependency issues
# Import only the providers you need
import anthropic

# Uncomment as needed for your providers
# import openai
# import mistralai.client
# from mistralai.client import MistralClient
# import groq

# Import tenacity for retries (if available)
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.logger.warning("Tenacity not available, retry functionality will be limited")


class LLMProvider:
    """
    Modular LLM Provider to support multiple LLM APIs with a unified interface.
    
    Supports multiple providers:
    - Anthropic Claude (default)
    - OpenAI GPT (optional)
    - Mistral AI (optional)
    - Groq (optional)
    
    Configuration happens through the config object which stores provider-specific
    details and handles credentials management.
    """
    
    def __init__(self, config):
        """Initialize the LLM provider with the specified configuration"""
        self.config = config
        self.provider_type = getattr(config, "LLM_PROVIDER", "anthropic").lower()
        
        # Get client model from config
        self.model = config.client_MODEL
        
        # Statistics tracking
        self.request_count = 0
        self.token_usage = 0
        self.last_request_time = 0
        self.min_request_interval = 60.0  # seconds between requests to avoid rate limits
        
        # Retry settings
        self.max_retries = 3
        self.retry_base_delay = 2  # seconds
        
        # Initialize the appropriate client based on provider type
        self._initialize_client()
        
        logger.logger.info(f"LLMProvider initialized with {self.provider_type.capitalize()} using model: {self.model}")
    
    def _initialize_client(self) -> None:
        """Initialize the appropriate client based on the provider type"""
        try:
            if self.provider_type == "anthropic":
                self.client = anthropic.Anthropic(api_key=self.config.client_API_KEY)
                logger.logger.debug("Anthropic Claude client initialized")
        
            elif self.provider_type == "openai":
                # Uncomment when using OpenAI
                # self.client = openai.OpenAI(api_key=self.config.client_API_KEY)
                # logger.logger.debug("OpenAI client initialized")
                logger.logger.warning("OpenAI support is commented out. Uncomment imports and client initialization to use.")
                self.client = None
        
            elif self.provider_type == "mistral":
                # Uncomment when using Mistral
                # self.client = MistralClient(api_key=self.config.client_API_KEY)
                # logger.logger.debug("Mistral AI client initialized")
                logger.logger.warning("Mistral AI support is commented out. Uncomment imports and client initialization to use.")
                self.client = None
        
            elif self.provider_type == "groq":
                # Uncomment when using Groq
                # self.client = groq.Client(api_key=self.config.client_API_KEY)
                # logger.logger.debug("Groq client initialized")
                logger.logger.warning("Groq support is commented out. Uncomment imports and client initialization to use.")
                self.client = None
            
            else:
                logger.logger.error(f"Unsupported LLM provider: {self.provider_type}")
                self.client = None
    
        except Exception as e:
            logger.log_error(f"Client Initialization Error ({self.provider_type})", str(e))
            self.client = None
    
    def _parse_prediction_text(self, text: str, token: str, timeframe: str, current_price: float) -> Dict[str, Any]:
        """
        Parse LLM-generated prediction text into structured data.
    
        Args:
            text: The raw prediction text from LLM
            token: The token symbol
            timeframe: Prediction timeframe
            current_price: Current token price
        
        Returns:
            Structured prediction dictionary
        """
        # Initialize with defaults - this ensures we always return valid structure
        prediction = self._get_fallback_prediction(token, timeframe, current_price)
    
        try:
            # Simple extraction of price target (looking for $ followed by number)
            import re
        
            # Try to extract price target
            price_matches = re.findall(r'\$([0-9,.]+)', text)
            if price_matches:
                try:
                    price_target = float(price_matches[0].replace(',', ''))
                    prediction["prediction"]["price"] = price_target
                    # Calculate percent change
                    if current_price > 0:
                        percent_change = ((price_target / current_price) - 1) * 100
                        prediction["prediction"]["percent_change"] = percent_change
                except (ValueError, IndexError):
                    pass
        
            # Try to extract confidence level
            confidence_matches = re.findall(r'(\d+)%\s+confidence', text.lower())
            if confidence_matches:
                try:
                    confidence = int(confidence_matches[0])
                    if 0 <= confidence <= 100:
                        prediction["prediction"]["confidence"] = confidence
                except (ValueError, IndexError):
                    pass
        
            # Try to extract price range
            range_matches = re.findall(r'\$([\d,.]+)\s*[-–]\s*\$([\d,.]+)', text)
            if range_matches:
                try:
                    lower = float(range_matches[0][0].replace(',', ''))
                    upper = float(range_matches[0][1].replace(',', ''))
                    prediction["prediction"]["lower_bound"] = lower
                    prediction["prediction"]["upper_bound"] = upper
                except (ValueError, IndexError):
                    pass
        
            # Extract sentiment
            if re.search(r'bullish|positive|upward|increase|gain|growth', text.lower()):
                prediction["sentiment"] = "BULLISH"
            elif re.search(r'bearish|negative|downward|decrease|drop|decline|fall', text.lower()):
                prediction["sentiment"] = "BEARISH"
            
            # Extract rationale (first couple of sentences after any numbers)
            rationale_match = re.search(r'([^.!?]*[.!?][^.!?]*[.!?])', text)
            if rationale_match:
                prediction["rationale"] = rationale_match.group(0).strip()
            
            return prediction
        
        except Exception as e:
            logger.logger.error(f"Error parsing prediction text: {str(e)}")
            # Return the default prediction instead of failing
            return prediction
    
    def _get_fallback_prediction(self, token: str, timeframe: str, current_price: float, price_change: float = 0.0) -> Dict[str, Any]:
        """
        Generate a fallback prediction when normal generation fails
    
        Args:
            token: Token symbol
            timeframe: Prediction timeframe
            current_price: Current token price
            price_change: Price change percentage (optional)
        
        Returns:
            Default structured prediction
        """
        # Calculate reasonable fallback values based on timeframe
        if timeframe == "1h":
            percent_change = max(0.5, min(1.0, abs(price_change) * 0.2)) * (1 if price_change >= 0 else -1)
            confidence = 60
            range_factor = 0.01
        elif timeframe == "24h":
            percent_change = max(1.0, min(2.0, abs(price_change) * 0.3)) * (1 if price_change >= 0 else -1)
            confidence = 55
            range_factor = 0.025  
        else:  # 7d
            percent_change = max(2.0, min(4.0, abs(price_change) * 0.4)) * (1 if price_change >= 0 else -1)
            confidence = 50
            range_factor = 0.05
        
        price_target = current_price * (1 + percent_change/100)
    
        return {
            "prediction": {
                "price": price_target,
                "confidence": confidence,
                "lower_bound": current_price * (1 - range_factor),
                "upper_bound": current_price * (1 + range_factor),
                "percent_change": percent_change,
                "timeframe": timeframe
            },
            "rationale": f"Technical analysis for {token} over {timeframe} timeframe suggests potential movement.",
            "sentiment": "BULLISH" if percent_change > 0 else "BEARISH" if percent_change < 0 else "NEUTRAL",
            "key_factors": ["Market trends", "Volume analysis", "Price action"]
        }
    
    def _clean_quotation_marks(self, text: str) -> str:
        """
        Clean quotation marks from the LLM response text
        
        This handles:
        1. Removing leading/trailing double quotes
        2. Removing leading/trailing single quotes
        3. Handling triple quotes that might appear in Python-style multiline strings
        4. Fixing ellipsis patterns that may appear as "..."
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text without unwanted quotation marks
        """
        if not text:
            return text
        
        # Remove leading/trailing triple quotes
        if text.startswith('"""') and text.endswith('"""'):
            text = text[3:-3]
        elif text.startswith("'''") and text.endswith("'''"):
            text = text[3:-3]
        
        # Remove leading/trailing double quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Remove leading/trailing single quotes
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Replace quoted phrases like "this" with this
        # But be careful not to affect apostrophes in contractions (don't, isn't, etc.)
        text = re.sub(r'"([^"]*)"', r'\1', text)
        text = re.sub(r"'([^']*)'", r'\1', text)
        
        # Replace triple dots with ellipsis character (optional)
        text = text.replace('...', '…')
        
        # Ensure we've removed any other obvious quotation patterns
        # This regex handles more complex patterns like quoted sentences within the text
        text = re.sub(r'(["\'])(.*?)(["\'])', r'\2', text)
        
        # Clean up any extra spaces that might have been introduced
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _make_api_call(self, **kwargs) -> Any:
        """
        Make API call with retry logic and proper error handling.
        
        Args:
            **kwargs: Provider-specific API call parameters
            
        Returns:
            API response or None if all retries failed
        """
        if not self.client:
            logger.logger.error(f"No initialized client for provider: {self.provider_type}")
            return None
            
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                if self.provider_type == "anthropic":
                    response = self.client.messages.create(**kwargs)
                    return response
                    
                # Add other providers as needed
                else:
                    logger.logger.error(f"API call not implemented for provider: {self.provider_type}")
                    return None
                    
            except anthropic.APIStatusError as e:
                # Handle API status errors with detailed logging
                status_code = getattr(e, 'status_code', None)
                error_response = getattr(e, 'response', None)
                request_id = error_response.headers.get('x-request-id') if error_response and hasattr(error_response, 'headers') else None
                
                logger.logger.error(
                    f"API Status Error: {status_code} - Provider: {self.provider_type}",
                    extra={
                        "request_id": request_id,
                        "status_code": status_code,
                        "error_type": getattr(e, 'type', 'unknown'),
                        "error_message": str(e)
                    }
                )
                
                # Don't retry 4xx errors as they're client-side and won't be resolved
                if status_code is not None and 400 <= status_code < 500:
                    return None
                
                # Only retry 5xx errors (server-side issues)
                if status_code is None or status_code < 500:
                    return None
                
                # Calculate backoff for retry
                backoff_time = self.retry_base_delay * (2 ** attempt)
                logger.logger.warning(f"Retrying API call in {backoff_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(backoff_time)
                continue
                
            except (anthropic.RateLimitError, anthropic.APITimeoutError) as e:
                logger.logger.warning(f"{e.__class__.__name__}: {str(e)}")
                
                # Calculate backoff for retry
                backoff_time = self.retry_base_delay * (2 ** attempt)
                logger.logger.warning(f"Retrying API call in {backoff_time}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(backoff_time)
                continue
                
            except Exception as e:
                logger.logger.error(f"Unexpected error during API call: {str(e)}", exc_info=True)
                return None
                
        # If we get here, all retries failed
        logger.logger.error(f"All {self.max_retries} API call attempts failed")
        return None
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, 
                     system_prompt: Optional[str] = None, preserve_json: bool = False) -> Optional[str]:
        """
        Generate text using the configured LLM provider
    
        Args:
            prompt: The user prompt or query
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
            system_prompt: Optional system prompt for providers that support it
            preserve_json: If True, skip quote cleaning to preserve JSON structure
        
        Returns:
            Generated text or None if an error occurred
        """
        if not self.client:
            logger.logger.error(f"No initialized client for provider: {self.provider_type}")
            return None
    
        self._enforce_rate_limit()
        self.request_count += 1
    
        try:
            # Provider-specific implementations
            if self.provider_type == "anthropic":
                messages = []
            
                # Add system message if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
            
                # Add user message
                messages.append({"role": "user", "content": prompt})
            
                # Make API call with error handling
                try:
                    # First try to make the API call
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages
                    )
                
                    # Validate the response to avoid NoneType errors
                    if response is None:
                        logger.logger.error("Received None response from API")
                        return None
                    
                    if not hasattr(response, 'content') or not response.content:
                        logger.logger.error("Invalid response format: missing content")
                        return None
                    
                    # Track approximate token usage
                    if hasattr(response, 'usage'):
                        self.token_usage += response.usage.output_tokens
                
                    # Safely extract the text content
                    try:
                        response_text = response.content[0].text
                    except (IndexError, AttributeError) as e:
                        logger.logger.error(f"Failed to extract text from response: {str(e)}")
                        return None
                
                    # Check if the response looks like JSON before applying cleaning
                    if preserve_json or self._looks_like_json(response_text):
                        # Skip quote cleaning for JSON to preserve structure
                        logger.logger.debug("JSON detected in response, preserving quotes")
                        # Just clean code blocks and remove leading/trailing quotes
                        clean_text = self._clean_json_response(response_text)
                    else:
                        # Apply standard quote cleaning for regular text
                        clean_text = self._clean_quotation_marks(response_text)
                    
                        # Log if quotation marks were found and cleaned (for debugging)
                        if response_text != clean_text:
                            logger.logger.debug("Quotation marks were cleaned from LLM response")
                
                    return clean_text
                
                except anthropic.APIStatusError as e:
                    # Handle API status errors specifically
                    status_code = getattr(e, 'status_code', None)
                    error_type = getattr(e, 'type', 'unknown')
                    error_message = str(e)
                
                    logger.logger.error(f"API Error {status_code} ({error_type}): {error_message}")
                
                    # If it's a 500 server error, we might want to retry
                    if status_code and status_code >= 500:
                        logger.logger.warning(f"Server error {status_code}, this may be transient")
                
                    return None
                
                except Exception as e:
                    logger.logger.error(f"Error during API call: {str(e)}")
                    return None
        
            elif self.provider_type == "openai":
                # Uncomment when using OpenAI
                """
                messages = []
            
                # Add system message if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
            
                # Add user message
                messages.append({"role": "user", "content": prompt})
            
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
                # Track token usage
                if hasattr(response, 'usage'):
                    self.token_usage += response.usage.completion_tokens
            
                response_text = response.choices[0].message.content
            
                # Check if the response looks like JSON before applying cleaning
                if preserve_json or self._looks_like_json(response_text):
                    # Skip quote cleaning for JSON to preserve structure
                    logger.logger.debug("JSON detected in response, preserving quotes")
                    clean_text = self._clean_json_response(response_text)
                else:
                    # Apply standard quote cleaning for regular text
                    clean_text = self._clean_quotation_marks(response_text)
                
                    # Log if quotation marks were found and cleaned (for debugging)
                    if response_text != clean_text:
                        logger.logger.debug("Quotation marks were cleaned from LLM response")
            
                return clean_text
                """
                logger.logger.warning("OpenAI generation code is commented out")
                return None
        
            elif self.provider_type == "mistral":
                # Uncomment when using Mistral
                """
                messages = []
            
                # Add system message if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
            
                # Add user message
                messages.append({"role": "user", "content": prompt})
            
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
                response_text = response.choices[0].message.content
            
                # Check if the response looks like JSON before applying cleaning
                if preserve_json or self._looks_like_json(response_text):
                    # Skip quote cleaning for JSON to preserve structure
                    logger.logger.debug("JSON detected in response, preserving quotes")
                    clean_text = self._clean_json_response(response_text)
                else:
                    # Apply standard quote cleaning for regular text
                    clean_text = self._clean_quotation_marks(response_text)
                
                    # Log if quotation marks were found and cleaned (for debugging)
                    if response_text != clean_text:
                        logger.logger.debug("Quotation marks were cleaned from LLM response")
            
                return clean_text
                """
                logger.logger.warning("Mistral AI generation code is commented out")
                return None
        
            elif self.provider_type == "groq":
                # Uncomment when using Groq
                """
                messages = []
            
                # Add system message if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
            
                # Add user message
                messages.append({"role": "user", "content": prompt})
            
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
                response_text = response.choices[0].message.content
            
                # Check if the response looks like JSON before applying cleaning
                if preserve_json or self._looks_like_json(response_text):
                    # Skip quote cleaning for JSON to preserve structure
                    logger.logger.debug("JSON detected in response, preserving quotes")
                    clean_text = self._clean_json_response(response_text)
                else:
                    # Apply standard quote cleaning for regular text
                    clean_text = self._clean_quotation_marks(response_text)
                
                    # Log if quotation marks were found and cleaned (for debugging)
                    if response_text != clean_text:
                        logger.logger.debug("Quotation marks were cleaned from LLM response")
            
                return clean_text
                """
                logger.logger.warning("Groq generation code is commented out")
                return None
        
            logger.logger.warning(f"Text generation not implemented for provider: {self.provider_type}")
            return None
        
        except Exception as e:
            logger.logger.error(f"{self.provider_type.capitalize()} API Error", str(e))
            return None
            
    def generate_json(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.5, 
                     system_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate JSON response using the configured LLM provider
        
        Args:
            prompt: The user prompt requesting JSON
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (lower for more deterministic JSON)
            system_prompt: Optional system prompt for providers that support it
            
        Returns:
            Parsed JSON dict or None if an error occurred
        """
        # Add explicit JSON formatting instructions if not already present
        if "JSON" not in prompt and "json" not in prompt:
            prompt = prompt + "\n\nPlease respond with valid JSON format only."
            
        # Use a lower temperature for more reliable JSON generation
        json_temperature = min(temperature, 0.5)
        
        # If no system prompt provided, add one that encourages valid JSON
        if not system_prompt:
            system_prompt = "You are a helpful assistant that responds with valid, properly formatted JSON. Always use double quotes for property names and string values according to the JSON specification."
        
        # Get the raw text response, preserving JSON structure
        response_text = self.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=json_temperature,
            system_prompt=system_prompt,
            preserve_json=True  # Important: preserve JSON structure
        )
        
        if not response_text:
            logger.logger.error("Failed to generate JSON response")
            return None
            
        # Attempt to parse the JSON
        try:
            # Clean any markdown code blocks or other non-JSON elements
            json_text = self._clean_json_response(response_text)
            json_data = json.loads(json_text)
            logger.logger.debug("Successfully parsed JSON response")
            return json_data
        except json.JSONDecodeError as e:
            logger.log_error("JSON Parsing", f"Failed to parse generated JSON: {e}")
            logger.logger.debug(f"Raw JSON text: {response_text[:200]}...")
            
            # Try to repair the JSON before giving up
            try:
                repaired_json = self._repair_json(response_text)
                json_data = json.loads(repaired_json)
                logger.logger.info("Successfully parsed JSON after repair")
                return json_data
            except Exception as repair_error:
                logger.log_error("JSON Repair", f"Failed to repair JSON: {repair_error}")
                return None
    
    def _looks_like_json(self, text: str) -> bool:
        """
        Check if the text appears to be JSON or contains JSON
    
        Args:
            text: The text to check
        
        Returns:
            True if the text looks like JSON, False otherwise
        """
        # Safety check for None
        if text is None:
            return False
        
        try:
            # Simple heuristic checks for JSON structure
            # 1. Starts with { and ends with }
            if (text.strip().startswith('{') and text.strip().endswith('}')) or \
               (text.strip().startswith('[') and text.strip().endswith(']')):
                return True
            
            # 2. Contains multiple key-value pairs with double quotes
            if re.search(r'"[^"]+"\s*:\s*("[^"]*"|[\d\[\{])', text):
                return True
            
            # 3. Check if the prompt explicitly requested JSON
            if '```json' in text or 'json' in text.lower() or 'JSON' in text:
                return True
            
            return False
        except Exception as e:
            logger.logger.debug(f"Error in _looks_like_json: {str(e)}")
            return False
    
    def _clean_json_response(self, text: str) -> str:
        """
        Clean a JSON response from markdown code blocks and other non-JSON elements
        while preserving the JSON structure
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned JSON text
        """
        # Safety check
        if text is None:
            return ""
            
        try:
            # Remove markdown code blocks if present
            if '```' in text:
                # Extract content from code blocks
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                if match:
                    text = match.group(1)
                    
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Remove leading/trailing triple quotes
            if text.startswith('"""') and text.endswith('"""'):
                text = text[3:-3].strip()
            elif text.startswith("'''") and text.endswith("'''"):
                text = text[3:-3].strip()
                
            # Remove single leading/trailing quotes that might wrap the entire JSON
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                # Only remove if they're wrapping the entire JSON object/array
                inner = text[1:-1].strip()
                if (inner.startswith('{') and inner.endswith('}')) or \
                   (inner.startswith('[') and inner.endswith(']')):
                    text = inner
                    
            return text
        except Exception as e:
            logger.logger.debug(f"Error in _clean_json_response: {str(e)}")
            return text
    
    def _repair_json(self, text: str) -> str:
        """
        Attempt to repair malformed JSON
        
        Args:
            text: The JSON text to repair
            
        Returns:
            Repaired JSON text
        """
        # Safety check
        if text is None:
            return "{}"
            
        try:
            # Clean any markdown or comment blocks first
            text = self._clean_json_response(text)
            
            # Replace JavaScript-style property names with proper JSON
            # This regex matches property names not in quotes and adds double quotes
            text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
            
            # Fix unquoted string values (specifically for known enum values)
            text = re.sub(r':\s*(BULLISH|BEARISH|NEUTRAL)([,}\s])', r': "\1"\2', text)
            text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)([,}\s])', r': "\1"\2', text)
            
            # Fix common array syntax issues
            text = re.sub(r'\[\s*,', '[', text)  # Remove leading commas in arrays
            text = re.sub(r',\s*\]', ']', text)  # Remove trailing commas in arrays
            
            # Fix object syntax issues
            text = re.sub(r'{\s*,', '{', text)   # Remove leading commas in objects
            text = re.sub(r',\s*}', '}', text)   # Remove trailing commas in objects
            
            # Fix double commas
            text = re.sub(r',\s*,', ',', text)
            
            # Fix missing commas between array elements
            text = re.sub(r'(true|false|null|"[^"]*"|[0-9.]+)\s+("|\{|\[|true|false|null|[0-9.])', r'\1, \2', text)
            
            # In case any single quotes were used instead of double quotes
            text = text.replace("'", '"')
            
            # Final catch-all for any remaining unquoted property names
            text = re.sub(r'([{,])\s*([^"\s{}\[\],]+)\s*:', r'\1"\2":', text)
            
            return text
        except Exception as e:
            logger.logger.debug(f"Error in _repair_json: {str(e)}")
            return "{}"
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embeddings for the provided text
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values or None if an error occurred
        """
        if not self.client:
            logger.logger.error(f"No initialized client for provider: {self.provider_type}")
            return None
        
        self._enforce_rate_limit()
        self.request_count += 1
        
        try:
            # Provider-specific embedding implementations
            if self.provider_type == "anthropic":
                # Note: Claude may not support embeddings directly
                logger.logger.warning("Embeddings not supported for Anthropic Claude")
                return None
            
            elif self.provider_type == "openai":
                # Uncomment when using OpenAI
                """
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",  # Use appropriate embedding model
                    input=text
                )
                return response.data[0].embedding
                """
                logger.logger.warning("OpenAI embeddings code is commented out")
                return None
            
            # Add other provider embedding implementations as needed
            
            logger.logger.warning(f"Embeddings not implemented for provider: {self.provider_type}")
            return None
            
        except Exception as e:
            logger.log_error(f"{self.provider_type.capitalize()} Embedding Error", str(e))
            return None
    
    def _enforce_rate_limit(self) -> None:
        """Simple rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the provider"""
        return {
            "provider": self.provider_type,
            "model": self.model,
            "request_count": self.request_count,
            "estimated_token_usage": self.token_usage,
            "last_request_time": self.last_request_time
        }
    
    def is_available(self) -> bool:
        """Check if the provider client is properly initialized and available"""
        return self.client is not None
        
    def generate_prediction(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate market prediction for a specific token with robust error handling.
        
        Args:
            token: The token symbol to generate a prediction for
            market_data: Current market data dictionary
            timeframe: Prediction timeframe (1h, 24h, 7d)
            
        Returns:
            Prediction dictionary or empty dict if generation failed
        """
        try:
            # Get token data
            token_data = market_data.get(token, {})
            if not token_data:
                logger.logger.error(f"Missing market data for token {token}")
                return {}
                
            # Extract required data for prediction prompt
            current_price = token_data.get('current_price', 0)
            price_change = token_data.get('price_change_percentage_24h', 0)
            
            # Prepare prediction prompt parameters
            prompt_params = {
                'token': token,
                'timeframe': timeframe,
                'current_price': current_price,
                'price_change': price_change,
                # Add other parameters needed for prediction prompt
                'trend': 'bullish' if price_change > 0 else 'bearish',
                'trend_strength': min(100, max(0, abs(price_change) * 10)),  # Scale 0-100
                'rsi': 50,  # Placeholder
                'macd_signal': 'positive' if price_change > 0 else 'negative',
                'bb_signal': 'neutral',
                'volatility': abs(price_change),
                'arima_price': current_price * (1 + price_change/200),
                'arima_lower': current_price * (1 - 0.01),
                'arima_upper': current_price * (1 + 0.02),
                'ml_price': current_price * (1 + price_change/150),
                'ml_lower': current_price * (1 - 0.015),
                'ml_upper': current_price * (1 + 0.025),
                'market_sentiment': 'bullish' if price_change > 3 else 'bearish' if price_change < -3 else 'neutral'
            }
            
            # Format the prediction prompt
            prediction_prompt = self.config.client_PREDICTION_PROMPT.format(**prompt_params)
            
            # Generate the prediction text
            prediction_text = self.generate_text(prediction_prompt, max_tokens=1000)
            
            # Handle failed generation
            if not prediction_text:
                logger.logger.error(f"Failed to generate prediction for {token} ({timeframe})")
                return self._get_fallback_prediction(token, timeframe, current_price, price_change)
                
            # Parse the prediction text into structured data
            try:
                # First try to parse as JSON - some responses might be formatted as JSON
                if self._looks_like_json(prediction_text):
                    try:
                        json_text = self._clean_json_response(prediction_text)
                        parsed_prediction = json.loads(json_text)
                        # Ensure required structure
                        if 'prediction' not in parsed_prediction:
                            parsed_prediction = self._parse_prediction_text(prediction_text, token, timeframe, current_price)
                        return parsed_prediction
                    except json.JSONDecodeError:
                        # Fall back to regex parsing
                        return self._parse_prediction_text(prediction_text, token, timeframe, current_price)
                else:
                    # Regular text parsing
                    return self._parse_prediction_text(prediction_text, token, timeframe, current_price)
            except Exception as e:
                logger.logger.error(f"Error parsing prediction for {token} ({timeframe}): {str(e)}")
                # Return fallback prediction
                return self._get_fallback_prediction(token, timeframe, current_price, price_change)
                
        except Exception as e:
            logger.logger.error

