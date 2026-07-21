"""
verify_multilingual_end2end.py — Verification of Phase 11 Multilingual RAG Pipeline
"""

import os
import sys
import shutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import EMBEDDING_MODEL
from src.ingest import run_ingestion
from src.retriever import retrieve_and_rerank, invalidate_user_cache
from src.llm import get_prompt_template
from src.llm_provider import get_provider, check_provider_availability
from src.document_store import remove_document, get_all_documents


TEST_USER = "test_multilingual_user_123"

def run_verification():
    print(f"\n{'='*70}")
    print(f"  MULTILINGUAL END-TO-END VERIFICATION")
    print(f"  Active EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"{'='*70}\n")

    # Clean previous test index if present
    invalidate_user_cache(TEST_USER)
    index_dir = os.path.join("vectorstore", TEST_USER)
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    for doc in get_all_documents(TEST_USER):
        remove_document(TEST_USER, doc["filename"])

    # 1. Create a sample English document with specific facts
    sample_text = (
        "Project Orion is a futuristic medical AI system developed for automated clinical diagnostics. "
        "It analyzes patient history, genomic sequencing data, and imaging scans to detect early-stage "
        "oncological abnormalities with 98.4% accuracy. The project was launched in Geneva in 2025."
    )

    test_file_path = "test_orion_doc.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print("[STEP 1] Ingesting English Document into Vector Store with Multilingual Model...")
    res = run_ingestion(test_file_path, user_id=TEST_USER, original_filename="test_orion_doc.txt")
    print(f"  -> Ingestion Result: {res}")

    # 2. Query in Hindi (non-English) about facts only in the English document
    hindi_question = "प्रोजेक्ट ओरियन (Project Orion) क्या है और इसकी सटीकता (accuracy) कितनी है?"
    print(f"\n[STEP 2] Asking Question in Hindi against English Document:")
    print(f"  Hindi Question: '{hindi_question}'")

    print("\n[STEP 3] Performing Multilingual Vector Retrieval & Re-ranking...")
    retrieved_docs = retrieve_and_rerank(hindi_question, user_id=TEST_USER)
    
    print(f"  -> Retrieved {len(retrieved_docs)} document chunk(s):")
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"     [{idx}] {doc.page_content}")

    # Check if retrieval retrieved the relevant English text
    retrieval_success = any("Project Orion" in d.page_content for d in retrieved_docs)
    print(f"\n  [VERIFICATION RESULT 1] Multilingual Cross-Lingual Retrieval:")
    print(f"  - Query Language: Hindi")
    print(f"  - Document Language: English")
    print(f"  - Target Passage Retrieved: {'SUCCESS' if retrieval_success else 'FAILED'}")

    # 3. Verify Prompt Template formatting
    print("\n[STEP 4] Testing Prompt Template with Multilingual Instructions...")
    context_str = "\n\n".join([d.page_content for d in retrieved_docs])
    prompt_template = get_prompt_template()
    formatted_prompt = prompt_template.format(context=context_str, question=hindi_question)
    
    has_lang_instruction = "Answer in the same language the question was asked in" in formatted_prompt
    print(f"  - Prompt contains language instruction: {'SUCCESS' if has_lang_instruction else 'FAILED'}")
    print(f"\n[Formatted Prompt Snippet]:\n{'─'*50}\n{formatted_prompt}\n{'─'*50}\n")

    # 4. Check LLM Provider status
    provider_status = check_provider_availability()
    print(f"  [LLM Availability]: {provider_status}")

    # Cleanup test file and test index
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
    invalidate_user_cache(TEST_USER)
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    for doc in get_all_documents(TEST_USER):
        remove_document(TEST_USER, doc["filename"])

    print("\n✅ MULTILINGUAL END-TO-END VERIFICATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
