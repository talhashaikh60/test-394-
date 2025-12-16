# HealthTrack Front Desk Policy Chatbot

This repository documents the recommended approach for a HealthTrack front desk chatbot that answers staff questions strictly from official policies.

## Why use Retrieval-Augmented Generation (RAG)?
- Answers are grounded in uploaded policy documents instead of guesswork.
- Responses stay consistent and repeatable as policies change.
- Clear fallback when information is missing: "I don’t see this covered in our policies. Please check with a manager."

Regular chatbots rely on general knowledge and can mix or invent policies. A RAG chatbot pulls answers directly from your documents, eliminating those risks.

## High-level flow
1. Upload policy documents (guest policies, membership rules, guest passes, pool rules, locker room and age restrictions, etc.).
2. LangChain chunks the documents and stores embeddings in a vector database.
3. When staff ask a question, the chatbot retrieves the most relevant chunks and answers **only** with that content.

## System prompt (copy-paste into LangChain)
```
You are HealthTrack’s Front Desk Policy Assistant.

Your job is to help front desk staff quickly and accurately answer common guest and member questions using ONLY the provided HealthTrack policy documents.

IMPORTANT RULES:
- Only answer questions using information found in the provided documents.
- Do NOT guess, assume, or invent policies.
- If the answer is not clearly stated in the documents, say:
  “I don’t see this covered in our policies. Please check with a manager.”
- Keep answers short, clear, and practical for real front desk conversations.
- Use simple, friendly language that staff can confidently say out loud to guests.
- If a policy has conditions (age limits, time restrictions, fees, exceptions), clearly list them.
- If multiple policies apply, summarize them cleanly in bullet points.

TONE:
- Professional but friendly
- Clear and direct
- No legal language
- No long explanations unless needed

WHEN ANSWERING:
- Prioritize accuracy over completeness
- If helpful, include:
  - Who the policy applies to
  - Any limits or exceptions
  - What staff should do next (allow, deny, escalate to manager)

EXAMPLES OF QUESTIONS YOU SHOULD HANDLE WELL:
- “Can a guest use the pool without a member?”
- “How many guests can a member bring?”
- “Are kids allowed in the locker room?”
- “What’s the guest pass policy?”
- “What do we do if someone forgot their ID?”
- “Can a guest use the facility multiple times in one day?”

If a question is unclear, ask ONE short clarifying question before answering.
```

## Optional enhancements
- End answers with **"Staff Action:"** guidance and the policy source when relevant.
- Flag high-risk situations (refunds, disputes, safety issues) for escalation.

## Next steps (actionable)
1. **Prepare the policy corpus**
   - Convert PDFs/Word docs to clean text with consistent headings ("Guest Passes", "Age Restrictions", etc.).
   - Normalize terminology (e.g., always use "guest pass" vs. "day pass") and remove outdated copies.
   - Store a per-section `policy_name` so the bot can cite or log where answers came from.

2. **Implement the RAG pipeline**
   - Chunk docs by heading (not character count alone) to keep policy context intact.
   - Index embeddings in a vector store (e.g., Chroma, PGVector, Pinecone) with `policy_name` metadata.
   - Use a retrieval-augmented chain with a strict system prompt (above) and `k=3-5` top chunks.
   - Add a second-stage filter that rejects answers if retrieved text is below a relevance threshold.

3. **Baseline LangChain skeleton (Python)**
   ```python
   from langchain_openai import ChatOpenAI, OpenAIEmbeddings
   from langchain.chains import RetrievalQA
   from langchain.vectorstores import Chroma

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

   # Assume documents are already chunked by heading with metadata={"policy_name": "Guest Passes"}
   vectordb = Chroma(persist_directory=".chroma", embedding_function=embeddings)
   retriever = vectordb.as_retriever(search_kwargs={"k": 4})

   qa = RetrievalQA.from_chain_type(
       llm=llm,
       retriever=retriever,
       chain_type="stuff",  # or map_rerank if answers get long
       chain_type_kwargs={"prompt": SYSTEM_PROMPT},
   )

   response = qa.invoke({"query": "Can a guest use the pool without a member?"})
   print(response["result"])
   ```
   - Use streaming + logging to capture which chunks were used for each answer.
   - Add a guardrail: if `qa` returns low confidence or no sources, reply with the manager escalation message.

4. **Test with real scenarios**
   - Build a small QA set from front desk tickets (10–20 common questions) and verify answers match the policies.
   - Include edge cases: expired memberships, minors with/without guardians, locker room access, multiple same-day visits.
   - Have staff trial the bot in shadow mode and log any "I don't know" responses to backfill the corpus.

5. **Operational guardrails**
   - Track which `policy_name` chunks were used for each answer for auditability.
   - Version policy documents and re-embed on every update; add a quick smoke test before deployment.
   - Limit output length (e.g., 80–120 words) and keep temperature at 0 to enforce consistency.
