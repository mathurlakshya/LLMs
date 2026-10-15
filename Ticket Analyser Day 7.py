import os
import time
from google import genai
from pydantic import BaseModel,Field
from typing import Literal
import logging

class TicketAnalysis(BaseModel):
    category: str = Field(description='Main topic of the message')
    sentiment: Literal['Positive','Negative','Neutral']
    urgency: Literal["high", "medium", "low"]
    summary: str = Field(description="One-line summary")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler('app.log',encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
start_time = time.perf_counter()

user_text = input('Enter your message: ').strip()
model = 'gemini-3.8-flash'
prompt = f"""
Analyse this message:
{user_text}

Avoid any introduction or conclusion, 
And return exactly 4 things:
1. category
2.sentiment (positive or negative)
3.urgency (high, medium or low)
4. One line summary of what user is asking or giving the message
"""    

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise SystemExit("GEMINI_API_KEY is missing!")
client = genai.Client(api_key=api_key)

max_retries = 3
for attempts in range(max_retries):
    try:
        logger.info('Sending request to gemini...')
        response = client.interactions.create(
            model = model,
            input = prompt,
            response_format = {
                'type': 'text',
                'mime_type': 'application/json',
                'schema': TicketAnalysis.model_json_schema(),
            }
        )
        elapsed = time.perf_counter() - start_time

        result = TicketAnalysis.model_validate_json(response.output_text)
        print("\n--- Ticket Analysis ---")
        print(result.model_dump_json(indent=2))

        logger.info("Request succeeded")
        logger.info("Request duration: %.2f seconds", elapsed)

        usage = getattr(response, "usage", None)

        if usage:
                input_tokens = getattr(usage, "total_input_tokens", None)
                output_tokens = getattr(usage, "total_output_tokens", None)
        
                logger.info("Input tokens: %s", input_tokens)
                logger.info("Output tokens: %s", output_tokens)
        
        else:
                logger.info("Token usage was not available in this response")
        break        

    except Exception as e:
        logger.exception("Request or validation failed")

        if attempts == max_retries - 1:
            print("All attempts failed. Check app.log.")
            break 
        
        wait_time = 2 ** attempts
        logger.info(
            "Attempt %d failed. Retrying in %d seconds.",
            attempts + 1,
            wait_time,
        )
        time.sleep(wait_time)    


        

