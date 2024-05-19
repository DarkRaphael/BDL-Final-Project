from fastapi import FastAPI, File, UploadFile, Request
#from tensorflow.keras.models import load_model
import numpy as np
#from PIL import Image
from io import BytesIO
import sys

import time


from prometheus_client import Summary, start_http_server, Counter, Gauge
from prometheus_client import disable_created_metrics

REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
counter = Counter('api_call_counter', 'number of times that API is called', ['endpoint', 'client'])
gauge = Gauge('api_runtime_secs', 'runtime of the method in seconds', ['endpoint', 'client'])


# Define the FastAPI app
app = FastAPI()

# Data model for request
#class TextData(BaseModel):
#    texts: List[str]

# Define the endpoint for sentiment analysis
@REQUEST_DURATION.time()
@app.post("/analyze-sentiment")
async def analyze_sentiment(request:Request, text: str):
    counter.labels(endpoint='/analyze-sentiment', client=request.client.host).inc()
    
    start = time.time()
    '''
    texts = data.textsS
    # Create a DataFrame from the input texts
    df = spark.createDataFrame([(text,) for text in texts], ["text"])

    # Process the DataFrame as per preprocessing pipeline
    df_cleaned = df.withColumn('cleaned_text', clean_text_udf(df['text']))
    df_tokenized = tokenizer.transform(df_cleaned)
    df_no_stopwords = remover.transform(df_tokenized)
    df_lemmatized = df_no_stopwords.withColumn('lemmatized_words', lemmatize_udf(df_no_stopwords['filtered_words']))
    df_features = hashingTF.transform(df_lemmatized)
    df_rescaled = idfModel.transform(df_features)

    # Make predictions using developed model
    predictions = model.transform(df_rescaled)
    results = predictions.select('prediction').collect()
    
    # Map predictions to sentiment labels
    sentiment_mapping = {-1: 'negative', 0: 'neutral', 1: 'positive'}  #Labels can be adjusted accordingly
    sentiments = [sentiment_mapping[int(row.prediction)] for row in results]
    '''

    for i in range (1000000):
    	x = 2
	
    time_taken = time.time() - start
    
    gauge.labels(endpoint='/analyze-sentiment', client=request.client.host).set(time_taken)
    

    return {"sentiment": "positive"}

if __name__ == '__main__':
    import uvicorn
    start_http_server(10000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
