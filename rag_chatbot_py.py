import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
# ---------- Groq client ----------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------- Knowledge base----------
knowledge_base = [
    {"emotion": "happy", "type": "music", "text": "Great energy! Try upbeat music like 'Good as Hell' by Lizzo to keep the momentum going."},
    {"emotion": "happy", "type": "activity", "text": "This is a great time to share your mood — reach out to someone or write down what's making you happy."},
    {"emotion": "sad", "type": "music", "text": "Listen to 'Here Comes the Sun' by The Beatles — a gentle, uplifting track that often helps shift a low mood."},
    {"emotion": "sad", "type": "activity", "text": "Step outside for a short walk, even 5-10 minutes. Sunlight and movement can genuinely help lift sadness."},
    {"emotion": "angry", "type": "music", "text": "Try calming instrumental or lo-fi music to help release tension without escalating the feeling."},
    {"emotion": "angry", "type": "activity", "text": "Try box breathing: inhale 4 seconds, hold 4, exhale 4, hold 4. Repeat 4-5 times."},
    {"emotion": "disgust", "type": "music", "text": "A calming, neutral playlist can help reset your mood after something unpleasant."},
    {"emotion": "disgust", "type": "activity", "text": "Step away from whatever triggered this feeling for a few minutes, and refocus on something you enjoy."},
    {"emotion": "surprise", "type": "music", "text": "Something upbeat or curious-sounding music can match this energetic moment."},
    {"emotion": "surprise", "type": "activity", "text": "Take a breath and give yourself a second to process what just happened."},
    {"emotion": "fear", "type": "music", "text": "Slow tempo ambient or nature sounds can help calm a racing mind."},
    {"emotion": "fear", "type": "activity", "text": "Try grounding: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste."},
    {"emotion": "neutral", "type": "activity", "text": "Take a moment to check in with yourself — how are you really feeling today?"},
]

# ---------- Embedder (sentence transformer) ----------
# Initialize the sentence transformer used to create embeddings for retrieval.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Retrieval ----------
def retrieve(detected_emotion, query=None, top_k=2):
    detected_emotion = detected_emotion.lower().strip()
    filtered = [item for item in knowledge_base if item["emotion"] == detected_emotion]
    if not filtered:
        filtered = [item for item in knowledge_base if item["emotion"] == "neutral"]
    texts = [item["text"] for item in filtered]
    text_embeddings = embedder.encode(texts)
    if query is None:
        return texts[:top_k]
    query_emb = embedder.encode([query])[0]
    scores = np.dot(text_embeddings, query_emb) / (
        np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in top_idx]

# ---------- Generation ----------
def rag_answer(detected_emotion, top_k=2):
    context_items = retrieve(detected_emotion, top_k=top_k)
    context = "\n".join(context_items)

    messages = [
        {"role": "system", "content": (
            "You are a warm, supportive companion. The user's detected emotion is "
            f"'{detected_emotion}'. Use the suggestions below naturally in your response, "
            "but always lead with empathy. Keep it short (2-4 sentences), conversational, "
            "and never clinical or robotic."
        )},
        {"role": "user", "content": f"Suggestions available:\n{context}\n\nRespond supportively."}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print(rag_answer("happy"))