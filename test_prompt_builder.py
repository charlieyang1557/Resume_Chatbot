#!/usr/bin/env python3
"""
Test prompt builder without heavy dependencies.
"""

from prompt_templates import PromptBuilder, PromptTemplates
from data_loader import load_corpus


def main():
    """Test prompt builder functionality."""
    print("🧪 Testing Prompt Builder")
    print("=" * 40)
    
    # Load some sample records
    print("📚 Loading sample records...")
    records = load_corpus("corpus_original.jsonl")
    
    # Find PTC Onshape records
    ptc_records = [r for r in records if "PTC" in r.text or "Onshape" in r.text]
    print(f"Found {len(ptc_records)} PTC/Onshape records")
    
    if not ptc_records:
        print("❌ No PTC/Onshape records found for testing")
        return
    
    # Create mock search results
    from retriever import SearchResult
    search_results = [SearchResult(record=r, score=0.95) for r in ptc_records[:2]]
    
    # Test different prompt templates
    templates = {
        "recruiter": PromptTemplates.recruiter_prompt(),
        "hiring_manager": PromptTemplates.hiring_manager_prompt(),
        "technical": PromptTemplates.technical_prompt(),
        "general": PromptTemplates.general_prompt()
    }
    
    test_question = "What did Charlie work on at PTC Onshape?"
    
    for template_name, system_prompt in templates.items():
        print(f"\n📝 Testing {template_name.upper()} template:")
        print("-" * 30)
        
        # Create prompt builder with specific template
        config = PromptBuilder()._get_default_config()
        config.system_prompt = system_prompt
        builder = PromptBuilder(config)
        
        # Build prompt
        prompt = builder.build_prompt(test_question, search_results)
        
        print(f"Prompt length: {len(prompt)} characters")
        print(f"System prompt preview: {system_prompt[:100]}...")
        print(f"Context preview: {search_results[0].record.text[:100]}...")
        
        # Test citation extraction
        test_response = f"Charlie worked on anomaly detection at [resume:Experience] and [website:Projects]."
        citations = builder.extract_citations(test_response)
        print(f"Citations extracted: {citations}")
        
        # Test response validation
        validation = builder.validate_response(test_response, has_context=True)
        print(f"Validation: {'✅ Safe' if validation['is_safe'] else '⚠️ Issues'}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
    
    # Test context formatting
    print(f"\n📄 Context Formatting Test:")
    print("-" * 30)
    
    builder = PromptBuilder()
    context_text = builder._build_context(search_results)
    print(f"Context length: {len(context_text)} characters")
    print(f"Context preview:\n{context_text[:300]}...")
    
    # Test sources extraction
    from retriever import DocumentRetriever
    retriever = DocumentRetriever(None)  # Mock retriever
    retriever.records = records  # Add records for testing
    
    sources = retriever.get_sources(search_results)
    print(f"\n📚 Sources extracted: {len(sources)}")
    for source in sources:
        print(f"  - {source['source']}:{source['section']} (score: {source['relevance_score']:.3f})")
    
    print(f"\n✅ Prompt builder test completed successfully!")


if __name__ == "__main__":
    main()
